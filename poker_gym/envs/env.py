from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple, Union, SupportsFloat, Literal, TypedDict

import numpy as np
from numpy import ndarray

from gymnasium.core import ActType, ObsType, RenderFrame

import poker_game
import gymnasium as gym
from gymnasium import spaces

from poker_game import poker
from poker_game.poker.engine import Dealer

from poker_gym import agent, error


class PokerEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
            self,
            num_players: Union[int, Literal["rnd"]],
            num_streets: int,
            blinds: Union[int, List[int]],
            antes: Union[int, List[int]],
            raise_sizes: Union[
                int, Literal["pot", "inf"], List[Union[int, Literal["pot", "inf"]]]
            ],
            num_raises: Union[int, Literal["inf"], List[Union[int, Literal["inf"]]]],
            num_suits: int,
            num_ranks: int,
            num_hole_cards: int,
            num_community_cards: Union[int, List[int]],
            num_cards_for_hand: int,
            mandatory_num_hole_cards: int,
            start_stack: int,
            low_end_straight: bool = True,
            order: Optional[List[str]] = None,
    ):
        super().__init__()
        self.action_space = spaces.MultiDiscrete([5,5])
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([10.0, 20.0]), dtype=np.float64)
        self.dealer = Dealer()

    def _get_obs(self) -> ObsDict:
        obs: ObsDict = {
            "action": self.dealer.action,
            "active": self.dealer.active,
            "button": self.dealer.button,
            "call": self.dealer.call,
            "community_cards": self.dealer.community_cards,
            # "hole_cards": self.hole_cards[self.action],
            "hole_cards": self.dealer.hole_cards,
            "max_raise": max_raise,
            "min_raise": min_raise,
            "pot": self.pot,
            "stacks": self.stacks,
            "street_commits": self.street_commits,
        }
        return obs

    def _get_info(self) -> dict:
        return {}

    def step(
            self, action: ActType
    ) -> tuple[ObsDict, SupportsFloat, bool, bool, dict[str, Any]]:
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
    ) -> tuple[ObsDict, dict[str, Any]]:
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        return None

    def close(self):
        pass


class ObsDict(TypedDict):
    action: int
    active: List[bool]
    button: int
    call: int
    community_cards: List[poker.Card]
    hole_cards: List[poker.Card]
    max_raise: int
    min_raise: int
    pot: int
    stacks: List[int]
    street_commits: List[int]


def register(configs: Dict[str, poker_game.configs.PokerConfig]) -> None:
    """Registers dict of poker_game configs as gym environments

    Parameters
    ----------
    configs : Dict
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
    for env_id, config in configs.items():
        gym.envs.registration.register(
            id=env_id, entry_point=env_entry_point, kwargs={**config}
        )
