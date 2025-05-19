import random

import poker_game
import pytest

import poker_gym


def test_base() -> None:
    agent = poker_gym.agent.BaseAgent()

    with pytest.raises(NotImplementedError):
        agent.act({})


def test_agent() -> None:
    agent = poker_gym.agent.kuhn.NashKuhnAgent(0)

    obs = {
        "action": 0,
        "active": [True, True],
        "button": 1,
        "call": 0,
        "community_cards": [],
        "hole_cards": [poker_game.Card("QS")],
        "max_raise": 1,
        "min_raise": 1,
        "pot": 2,
        "stacks": [9, 9],
        "street_commits": [0, 0],
    }
    assert agent.act(obs) == 0

    obs["hole_cards"] = [poker_game.Card("KS")]
    assert agent.act(obs) == 0

    obs["hole_cards"] = [poker_game.Card("AS")]
    assert agent.act(obs) == 0

    obs["pot"] = 4
    obs["hole_cards"] = [poker_game.Card("QS")]
    assert agent.act(obs) == 0

    obs["hole_cards"] = [poker_game.Card("KS")]
    random.seed(0)
    assert agent.act(obs) == 0
    random.seed(1)
    assert agent.act(obs) == 1

    obs["hole_cards"] = [poker_game.Card("AS")]
    assert agent.act(obs) == 1

    obs["action"] = ["1"]
    obs["pot"] = 2
    obs["hole_cards"] = [poker_game.Card("QS")]
    random.seed(0)
    assert agent.act(obs) == 0
    random.seed(1)
    assert agent.act(obs) == 1

    obs["hole_cards"] = [poker_game.Card("KS")]
    assert agent.act(obs) == 0

    obs["hole_cards"] = [poker_game.Card("AS")]
    assert agent.act(obs) == 1

    obs["pot"] = 3
    obs["hole_cards"] = [poker_game.Card("QS")]
    assert agent.act(obs) == 0

    obs["hole_cards"] = [poker_game.Card("KS")]
    random.seed(0)
    assert agent.act(obs) == 0
    random.seed(1)
    assert agent.act(obs) == 1

    obs["hole_cards"] = [poker_game.Card("AS")]
    assert agent.act(obs) == 1
