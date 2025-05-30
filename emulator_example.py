from examples.players import random_player, fish_player
from pypokerengine.engine.player import Player

from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator


class OneActionModel(BasePokerPlayer):
    FOLD, CALL, MIN_RAISE, MAX_RAISE = 0, 1, 2, 3


    def __init__(self, action):
        super().__init__()
        self.action = action

    def declare_action(self, valid_actions, hole_card, round_state):
        if self.FOLD == self.action:
            return valid_actions[0]['action'], valid_actions[0]['amount']
        elif self.CALL == self.action:
            return valid_actions[1]['action'], valid_actions[1]['amount']
        elif self.MIN_RAISE == self.action:
            return valid_actions[2]['action'], valid_actions[2]['amount']['min']
        elif self.MAX_RAISE == self.action:
            return valid_actions[2]['action'], valid_actions[2]['amount']['max']
        else:
            raise Exception("Invalid action [ %s ] is set" % self.action)


emulator = Emulator()
emulator.set_game_rule(3, 100, 1, 0)

ai_name = "0_ai"

players_info = {
    "1_rnd": {"name": "p_RND_1", "stack": 100},
    "2_rnd": {"name": "p_RND_2", "stack": 100},
    ai_name: {"name": "p_AI", "stack": 100},
}

emulator.register_player(list(players_info)[0], random_player.RandomPlayer())
emulator.register_player(list(players_info)[1], random_player.RandomPlayer())

game_state = emulator.generate_initial_game_state(players_info)
game_state, events = emulator.start_new_round(game_state)
game_state, events = emulator.run_until_round_finish_with_player_ask(game_state, ai_name)
print(game_state, events)

