from __future__ import annotations

import sys
from enum import Enum

import numpy as np
import gymnasium as gym

from gymnasium.core import ActType, RenderFrame, ObsType
from gymnasium import spaces

from typing import Any, Dict, List, Optional, Tuple, Union, SupportsFloat, Literal, TypedDict

from gymnasium.spaces import Discrete, Box, MultiDiscrete

from poker_gym import agent, error, configs


# region Classes and Enums
class Actions(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET_25 = 3
    BET_33 = 4
    BET_50 = 5
    BET_80 = 6
    BET_150 = 7
    RAISE_3BET = 8
    RAISE_3POT = 9
    ALL_IN = 10
    # Perhaps
    SMALL_BLIND = 11
    BIG_BLIND = 12


class Stages(Enum):
    PREFLOP = 0,
    FLOP = 1,
    TURN = 2,
    RIVER = 3,
    SHOWDOWN = 4,


class TableData(TypedDict):
    small_blind: float
    big_blind: float
    player_count: int
    start_stack: float


class PlayerData(TypedDict):
    position: bool
    stack: float
    hole_cards: List[Tuple[int]]


class CommunityData(TypedDict):
    position: List[bool]
    stage: List[bool]
    community_pot: float
    current_round_pot: float
    active_players: List[bool]
    community_cards: List[Tuple[int]]


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
    stage_data: np.ndarray


# endregion


MAX_PLAYER_COUNT = 10


class PokerEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            player_count: Union[int, Literal["rnd"]],
            blinds: List[int],
            start_stack: int,
    ):
        super().__init__()

        if player_count == "rnd":
            player_count = np.random.randint(2, MAX_PLAYER_COUNT + 1)
        else:
            player_count = int(player_count)

        self.table_data = TableData(small_blind=blinds[0],
                                    big_blind=blinds[1],
                                    player_count=player_count,
                                    start_stack=start_stack,)

        self.player_data = PlayerData(position=False,
                                      stack=start_stack,
                                      hole_cards=[])

        self.community_data = CommunityData(position=[False]*MAX_PLAYER_COUNT,
                                            stage=[False]*(len(Stages) - 1),
                                            active_players=[False]*MAX_PLAYER_COUNT,
                                            community_cards=[],
                                            community_pot=0,
                                            current_round_pot=0,)

        stage_data = StageData(calls=[False]*MAX_PLAYER_COUNT,
                               raises=[False]*MAX_PLAYER_COUNT,
                               min_calls_at_action=[0.]*MAX_PLAYER_COUNT,
                               contributions=[0.]*MAX_PLAYER_COUNT,
                               stack_at_action=[0.]*MAX_PLAYER_COUNT,
                               community_pot_at_action=[0.]*MAX_PLAYER_COUNT,)

        self.stage_data = [stage_data] * (len(Stages) - 1)

        self.obs = self._get_obs()
        self.action_space = spaces.Discrete(len(Actions) - 2)
        self.observation_space = spaces.Dict({
            "table_data": MultiDiscrete([0, self._get_table_data().size]),
            "player_data": MultiDiscrete([0, self._get_player_data().size]),
            "community_data": MultiDiscrete(np.array([0, self._get_community_data().size])),
            "stage_data": MultiDiscrete(np.array([[0, self._get_stage_data().shape[0]], [0, self._get_stage_data().shape[1]]])),
        })

    # region Private methods
    @staticmethod
    def _get_info() -> dict:
        return {}

    def _get_obs(self) -> ObservationData:
        obs: ObservationData = {
            "table_data": self._get_table_data(),
            "player_data": self._get_player_data(),
            "community_data": self._get_community_data(),
            "stage_data": self._get_stage_data(),
        }
        return obs

    def _get_table_data(self) -> np.ndarray:
        return np.zeros(len(self.table_data))

    def _get_player_data(self) -> np.ndarray:
        return np.zeros(len(self.player_data))

    def _get_community_data(self) -> np.ndarray:
        return np.zeros(len(self.community_data))

    def _get_stage_data(self) -> np.ndarray:
        return np.zeros(shape=(len(Stages) - 1, len(self.stage_data[0])))

    # endregion

    # region Implemented methods
    def step(
            self, action: ActType
    ) -> tuple[ObservationData, SupportsFloat, bool, bool, dict[str, Any]]:
        terminated = False
        truncated = False
        reward = np.random.random()
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, truncated, terminated, info

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> tuple[ObservationData, dict[str, Any]]:
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        return None

    def close(self):
        pass

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
