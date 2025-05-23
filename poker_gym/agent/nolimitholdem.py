import random
from poker_gym.agent import base


class NoLimitHoldemAgent(base.BaseAgent):

    def __init__(self, model: None) -> None:
        super().__init__()
        self.model = model

    # def act(self, obs: poker_game.poker.old_engine.ObservationDict) -> int:
    #     return -1
