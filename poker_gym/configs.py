import sys
from typing import List, Optional, Union, TypedDict, Literal, Any


class PokerConfig(TypedDict):
    small_blind: float
    start_stack: float


NO_LIMIT_HOLDEM_RANDOM_PLAYER: PokerConfig = {
    "small_blind": 1,
    "start_stack": 100,
}


# NO_LIMIT_HOLDEM_NINE_PLAYER: PokerConfig = {
#     "player_count": 9,
#     "blinds": [1, 2],
#     "start_stack": 100,
# }
#
# NO_LIMIT_HOLDEM_RANDOM_PLAYER: PokerConfig = {
#     "player_count": "rnd",
#     "blinds": [1, 2],
#     "start_stack": 100,
# }
