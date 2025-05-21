from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple, Union, SupportsFloat

import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame

import poker_game
import gymnasium as gym
from gymnasium import spaces

from poker_gym import agent, error


class PokerEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def _get_obs(self) -> ObsType:
        pass

    def _get_info(self) -> dict:
        pass

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
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
    ) -> tuple[ObsType, dict[str, Any]]:
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        return None


class ObservationWrapper(ObsType):
    def __init__(self):
        pass


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
