import poker_game


class BaseAgent:
    def __init__(self) -> None:
        pass

    def act(self, obs: poker_game.poker.engine.ObservationDict) -> int:
        raise NotImplementedError()
