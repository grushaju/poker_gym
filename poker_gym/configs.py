import sys
from typing import List, Optional, Union, TypedDict, Literal


class PokerConfig(TypedDict):
    player_count: Union[int, Literal["rnd"]]
    blinds: Union[int, List[int]]
    start_stack: int


NO_LIMIT_HOLDEM_NINE_PLAYER: PokerConfig = {
    "player_count": 9,
    "blinds": [1, 2],
    "start_stack": 100,
}

NO_LIMIT_HOLDEM_RANDOM_PLAYER: PokerConfig = {
    "player_count": "rnd",
    "blinds": [1, 2],
    "start_stack": 100,
}
