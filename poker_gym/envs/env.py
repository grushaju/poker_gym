from __future__ import annotations

import numpy as np
import gymnasium as gym

from gymnasium.core import ActType, RenderFrame, ObsType
from gymnasium import spaces

from typing import Any, Dict, List, Optional, Union, SupportsFloat, Literal, TypedDict, Iterable

from gymnasium.spaces import Box
from pypokerengine.engine.player import Player
from pypokerengine.api.emulator import Emulator
from pypokerengine.engine import Card

from poker_gym.agent import NoLimitHoldemAgent
from poker_gym.common import Stages, Actions, get_valid_action
from poker_gym.common import FOLD, CALL, RAISE, AMOUNT, MIN_AMOUNT, MAX_AMOUNT
from poker_gym import agent, error, configs
from pypokerengine.engine.poker_constants import PokerConstants as Const

MAX_PLAYER_COUNT = 10


# region Classes

class TableData(TypedDict):
    small_blind: float
    big_blind: float
    player_count: int
    start_stack: float


class PlayerData(TypedDict):
    position: bool
    stack: float
    hole_cards: List[List[int]]


class CommunityData(TypedDict):
    position: List[bool]
    stage: List[bool]
    community_pot: float
    current_round_pot: float
    active_players: List[bool]
    community_cards: List[List[int]]


class StageData(TypedDict):
    calls: List[bool]
    raises: List[bool]
    min_calls_at_action: List[float]
    contributions: List[float]
    stack_at_action: List[float]
    community_pot_at_action: List[float]


class ObservationData(TypedDict):
    table_data: np.ndarray
    player_data: np.ndarray
    community_data: np.ndarray
    pre_flop_data: np.ndarray
    flop_data: np.ndarray
    turn_data: np.ndarray
    river_data: np.ndarray


# endregion


# region Common methods
def one_hot(size, idx) -> np.ndarray:
    return np.eye(size)[idx].astype(dtype=np.float32)


def cards_to_list(cards: List[int]) -> List[List[int]]:
    if len(cards) == 0:
        return [[-1, -1]] * 5
    idx = []
    for card in cards:
        suit_pos = len(format(Card.get_suit_int(card), 'b')) - 1
        rank_pos = 13 - Card.get_rank_int(card) - 1
        idx += [[rank_pos, suit_pos]]
    return idx


def cards_to_one_hot(cards, size=-1):
    arr = []
    if size > len(cards):
        cards += [[-1, -1]] * (size - len(cards))
    for card in cards:
        if card[0] == card[1] == -1:
            arr = np.hstack((arr, np.zeros(13).astype(dtype=np.float32), np.zeros(4).astype(dtype=np.float32)))
        else:
            arr = np.hstack((arr, one_hot(13, card[0]), one_hot(4, card[1])))
    return arr


def get_stage_data(current_round, seats, main_pot, start_stack) -> dict:
    res = dict()
    res["calls"] = [False] * MAX_PLAYER_COUNT
    res["raises"] = [False] * MAX_PLAYER_COUNT
    res["min_calls_at_action"] = [0] * MAX_PLAYER_COUNT
    res["contribution"] = [0] * MAX_PLAYER_COUNT
    res["stack_at_action"] = [0] * MAX_PLAYER_COUNT
    res["community_pot_at_action"] = [0] * MAX_PLAYER_COUNT
    for item in current_round :
        if item["action"] == "FOLD":
            continue
        action, uuid, amount = item["action"], item["uuid"], item["amount"]
        idx, player = [(index, item) for (index, item) in enumerate(seats.players) if item.uuid == uuid][0]
        res["calls"][idx] = action == Player.ACTION_CALL_STR
        res["raises"][idx] = action == Player.ACTION_RAISE_STR
        res["contribution"][idx] += amount
        res["min_calls_at_action"][idx] = max(res["min_calls_at_action"][idx], amount)
        res["stack_at_action"][idx] = player.stack / MAX_PLAYER_COUNT / start_stack
        res["community_pot_at_action"][idx] = main_pot / MAX_PLAYER_COUNT / start_stack
    return res


def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def to_ndarray(items, size=-1) -> np.ndarray:
    arr = list(items)
    if size > len(arr):
        arr += [0] * (size - len(arr))
    return np.array(arr).astype(dtype=np.float32)


def get_info() -> dict:
    return {}


# endregion


class PokerEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            small_blind: int,
            start_stack: int
    ):
        super().__init__()

        self.uuid = "p_trained_uuid"
        self.emulator = None
        self.events = []
        self.last_game_state = {}

        # region Configure table data
        self.table_data = TableData(small_blind=small_blind,
                                    big_blind=small_blind * 2,
                                    player_count=MAX_PLAYER_COUNT,
                                    start_stack=start_stack, )
        low = np.array(
            [
                0,
                0,
                2,
                0
            ]
        ).astype(np.float32)
        high = np.array(
            [
                start_stack // 2,
                start_stack,
                MAX_PLAYER_COUNT,
                start_stack
            ]
        ).astype(np.float32)
        table_data_space = Box(low, high)
        # endregion

        # region Configure player data
        self.player_data = PlayerData(stack=1 / MAX_PLAYER_COUNT,
                                      hole_cards=[],
                                      position=False,
                                      )
        low = np.zeros(1 + 17 * 2 + 1).astype(np.float32)
        high = np.ones(1 + 17 * 2 + 1).astype(np.float32)
        player_data_space = Box(low, high)
        # endregion

        # region Configure community data
        self.community_data = CommunityData(stage=[False] * (len(Stages)),
                                            community_cards=[],
                                            community_pot=0,
                                            current_round_pot=0,
                                            position=[False] * MAX_PLAYER_COUNT,
                                            active_players=[False] * MAX_PLAYER_COUNT,
                                            )
        low = np.zeros(4 + 17 * 5 + 1 + 1 + MAX_PLAYER_COUNT + MAX_PLAYER_COUNT).astype(np.float32)
        high = np.ones(4 + 17 * 5 + 1 + 1 + MAX_PLAYER_COUNT + MAX_PLAYER_COUNT).astype(np.float32)
        community_data_space = Box(low, high)
        # endregion

        # region Configure stage data

        init_stage_data = StageData(calls=[False] * MAX_PLAYER_COUNT,
                                    raises=[False] * MAX_PLAYER_COUNT,
                                    min_calls_at_action=[0.] * MAX_PLAYER_COUNT,
                                    contributions=[0.] * MAX_PLAYER_COUNT,
                                    stack_at_action=[0.] * MAX_PLAYER_COUNT,
                                    community_pot_at_action=[0.] * MAX_PLAYER_COUNT, )

        self.pre_flop_stage_data = init_stage_data
        self.flop_stage_data = init_stage_data
        self.turn_stage_data = init_stage_data
        self.river_stage_data = init_stage_data

        low = np.zeros(MAX_PLAYER_COUNT * 6).astype(np.float32)
        high = np.ones(MAX_PLAYER_COUNT * 6).astype(np.float32)
        pre_flop_data_space = Box(low, high)
        flop_data_space = Box(low, high)
        turn_data_space = Box(low, high)
        river_data_space = Box(low, high)

        # endregion

        self.obs = self._get_obs()
        self.action_space = spaces.Discrete(len(Actions) - 2)
        self.observation_space = spaces.Dict({
            "table_data": table_data_space,
            "player_data": player_data_space,
            "community_data": community_data_space,
            "pre_flop_data": pre_flop_data_space,
            "flop_data": flop_data_space,
            "turn_data": turn_data_space,
            "river_data": river_data_space,
        })

    # region Private methods

    def _get_obs(self) -> ObservationData:
        obs: ObservationData = {
            "table_data": self._get_table_data(),
            "player_data": self._get_player_data(),
            "community_data": self._get_community_data(),
            "pre_flop_data": self._get_pre_flop_data(),
            "flop_data": self._get_flop_data(),
            "turn_data": self._get_turn_data(),
            "river_data": self._get_river_data(),
        }
        return obs

    def _get_table_data(self) -> np.ndarray:
        arr = to_ndarray(flatten(self.table_data.values()))
        return arr

    def _get_player_data(self) -> np.ndarray:
        arr = np.hstack(
            (np.array(self.player_data["stack"]).astype(np.float32),
             cards_to_one_hot(self.player_data["hole_cards"], 2),
             np.array(self.player_data["position"]).astype(np.float32)
             )
        )
        return arr

    def _get_community_data(self) -> np.ndarray:
        st, cards, pot, c_pot, pos, act = (
            to_ndarray(self.community_data["stage"]),
            cards_to_one_hot(self.community_data["community_cards"], 5),
            np.array(self.community_data["community_pot"]),
            np.array(self.community_data["current_round_pot"]),
            to_ndarray(self.community_data["position"], MAX_PLAYER_COUNT),
            to_ndarray(self.community_data["active_players"], MAX_PLAYER_COUNT)
        )
        arr = np.hstack((st, cards, pot, c_pot, pos, act))
        return arr

    def _get_pre_flop_data(self) -> np.ndarray:
        arr = to_ndarray(flatten(self.pre_flop_stage_data.values()))
        return arr

    def _get_flop_data(self) -> np.ndarray:
        arr = to_ndarray(flatten(self.flop_stage_data.values()))
        return arr

    def _get_turn_data(self) -> np.ndarray:
        arr = to_ndarray(flatten(self.turn_stage_data.values()))
        return arr

    def _get_river_data(self) -> np.ndarray:
        arr = to_ndarray(flatten(self.river_stage_data.values()))
        return arr

    # endregion

    # region Implemented methods
    def step(
            self, action: ActType
    ) -> tuple[ObservationData, SupportsFloat, bool, bool, dict[str, Any]]:
        terminated = False
        truncated = False
        action = Actions(action)

        game_state, events = self.emulator.run_until_game_finish_with_player_asking(
            self.last_game_state, self.uuid, self.update_obs_call)

        if game_state["street"] == Const.Street.FINISHED:
            pass

        next_player_pos = game_state["next_player"]
        valid_actions, hole_card, round_state = (
            self.emulator.get_state_before_play(next_player_pos, game_state))
        act, bet_amount, reward = get_valid_action(action, valid_actions, round_state)
        game_state, events = self.emulator.apply_action(game_state, act, bet_amount)
        self._update_obs(game_state, events)


        # next_player_pos = len(self.last_game_state["table"].seats.players) - 1
        # valid_actions, hole_card, round_state = (
        #     self.emulator.get_state_before_play(next_player_pos, self.last_game_state))
        # act, bet_amount, reward = get_valid_action(action, valid_actions, round_state)

        # game_state, events = self.emulator.apply_action(self.last_game_state, act, bet_amount)
        # self._update_obs(game_state, events)
        # game_state, events = self.emulator.run_until_ask_player(game_state, self.uuid, self.update_obs_call)
        # self._update_obs(game_state, events)
        # if game_state["street"] == Const.Street.FINISHED:
        #     game_state, events = self.emulator.run_until_ask_player(game_state, self.uuid, self.update_obs_call)
        self.last_game_state = game_state
        if game_state["street"] == Const.Street.FINISHED:
            if len([1 for p in game_state["table"].seats.players if p.is_active()]) == 1:
                self.emulator.generate_game_result_event(game_state)
        observation = self._get_obs()
        info = get_info()
        return observation, reward, truncated, terminated, info

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> tuple[ObservationData, dict[str, Any]]:

        player_count = options["player_count"] if options is not None else "rnd"
        if player_count == "rnd":
            player_count = np.random.randint(2, MAX_PLAYER_COUNT + 1)
        else:
            player_count = int(player_count)
        max_round = options["max_round"] if options is not None else 100
        model = options["model"] if options is not None else None

        self.events = []
        self.emulator = Emulator()
        self.emulator.set_game_rule(
            player_num=player_count,
            max_round=max_round,
            small_blind_amount=self.table_data["small_blind"],
            ante_amount=0
        )

        players_info = {}
        for i in range(player_count - 1):
            uuid = "old_agent_{}".format(i)
            player = NoLimitHoldemAgent(uuid, model, self.get_obs_call())
            players_info[uuid] = {"name": "p_old_{}".format(i), "stack": self.table_data["start_stack"]}
            self.emulator.register_player(uuid, player)
        self.uuid = "p_trained_uuid"
        players_info[self.uuid] = {"name": "p_trained", "stack": self.table_data["start_stack"]}

        game_state = self.emulator.generate_initial_game_state(players_info)
        game_state, events = self.emulator.start_new_round(game_state)
        # self.events += events
        self._update_obs(game_state, events)
        # game_state, events = self.emulator.run_until_ask_player(game_state, self.uuid, self.update_obs_call)
        # self._update_obs(game_state, events)

        observation = self._get_obs()
        info = get_info()
        self.last_game_state = game_state
        return observation, info

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        return None

    def close(self):
        pass

    # endregion

    # region Public methods
    def _update_obs(self, game_state, events):
        table = game_state["table"]
        self_player = table.seats.players[-1]
        self.player_data["stack"] = self_player.stack / self.table_data["start_stack"] / MAX_PLAYER_COUNT
        self.player_data["position"] = table.seats.players[table.dealer_btn].uuid == self_player.uuid
        self.player_data["hole_cards"] = cards_to_list(self_player.hole_card)

        ask_events = [item for item in events if item["type"] == "event_ask_player"]
        if len(ask_events) != 1:
            return
        round_state = ask_events[0]["round_state"]
        current_round = round_state["action_histories"][round_state["street"]]
        current_round_pot = sum(
            [item["amount"] for (index, item) in enumerate(current_round) if item["action"] != "FOLD"])
        main_pot = round_state["pot"]["main"]["amount"]
        side_pots = 0 \
            if len(round_state["pot"]["side"]) == 0 \
            else sum([side["amount"] for side in round_state["pot"]["side"]])
        self.community_data["community_cards"] = cards_to_list(table.get_community_card())
        self.community_data["community_pot"] = main_pot + side_pots
        self.community_data["current_round_pot"] = current_round_pot
        self.community_data["stage"] = [
            round_state["street"] == "river",
            round_state["street"] == "turn",
            round_state["street"] == "flop",
            round_state["street"] == "preflop"
        ]

        self.community_data["active_players"] = [item.pay_info.status != 2 for (index, item) in
                                                 enumerate(table.seats.players)]
        self.community_data["position"] = [index == table.dealer_btn for (index, item) in
                                           enumerate(table.seats.players)]

        if round_state["street"] == "preflop":
            self.pre_flop_stage_data = get_stage_data(current_round, table.seats, main_pot,
                                                      self.table_data["start_stack"])
        elif round_state["street"] == "flop":
            self.flop_stage_data = get_stage_data(current_round, table.seats, main_pot, self.table_data["start_stack"])
        elif round_state["street"] == "turn":
            self.turn_stage_data = get_stage_data(current_round, table.seats, main_pot, self.table_data["start_stack"])
        elif round_state["street"] == "river":
            self.river_stage_data = get_stage_data(current_round, table.seats, main_pot, self.table_data["start_stack"])

    def update_obs_call(self, game_state, events):
        self._update_obs(game_state, events)

    def get_obs_call(self):
        return lambda: self._get_obs()

    # endregion


def register(confs: Dict[str, configs.PokerConfig]) -> None:
    """Registers dict of poker_game configs as gym environments

    Parameters
    ----------
    confs : Dict
        dictionary of poker_game configs, keys must environment ids and
        values valid poker_game configs, example:
            configs = {
                'NoLimitHoldemNinePlayer-v0': {
                    'num_players': 9,
                    'num_streets': 4,
                    'blinds': [1, 2],
                    'antes': 0,
                    'raise_sizes': float('inf'),
                    'num_raises': float('inf'),
                    'num_suits': 4,
                    'num_ranks': 13,
                    'num_hole_cards': 2,
                    'num_community_cards': [0, 3, 1, 1],
                    'num_cards_for_hand': 5,
                    'mandatory_num_hole_cards': 0,
                    'start_stack': 200
                }
            }
    """
    env_entry_point = "poker_gym.envs.env:PokerEnv"
    for env_id, config in confs.items():
        gym.envs.registration.register(
            id=env_id, entry_point=env_entry_point, kwargs={**config}
        )
