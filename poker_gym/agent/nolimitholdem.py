
from stable_baselines3.common.base_class import BaseAlgorithm

from poker_gym.common import Actions, get_valid_action
from pypokerengine.players import BasePokerPlayer


class NoLimitHoldemAgent(BasePokerPlayer):

    def __init__(self, uuid: str, model: BaseAlgorithm, get_obs_func) -> None:
        super().__init__()
        self.uuid = uuid
        self.model = model
        self.get_obs_func = get_obs_func

    def declare_action(self, valid_actions, hole_card, round_state):
        obs = self.get_obs_func()
        action, _states = self.model.predict(obs)

        # action = Actions(action.item())
        action = Actions.CALL
        action, amount, __ = get_valid_action(action, valid_actions, round_state)
        return action, amount
