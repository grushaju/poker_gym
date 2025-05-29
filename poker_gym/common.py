from enum import Enum
from typing import List, Dict


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
    PRE_FLOP = 0,
    FLOP = 1,
    TURN = 2,
    RIVER = 3,


def get_action_values(actions: List[Dict[str, Dict]], value: str) -> tuple:
    fold = call = bet = None
    for index, item in enumerate(actions):
        val = item["action"]
        if val == FOLD:
            fold = item
        elif val == CALL:
            call = item
        elif val == RAISE:
            bet = item
    return fold, call, bet


FOLD, CALL, RAISE, AMOUNT, MIN_AMOUNT, MAX_AMOUNT = "fold", "call", "raise", "amount", "min", "max"


def get_valid_action(action, valid_actions: List[Dict[str, Dict]], round_state) -> (str, float, float):
    r_only = any(act["action"] == "RAISE" for act in round_state["action_histories"][round_state["street"]])
    reward = 0
    wrong_choose_reward = -0.1  # Wrong choose: -0.1
    right_choose_reward = 0.05  # Right choose: +0.05
    aggressive_bonus = 0.01  # Additional reward for betting
    act = FOLD
    amount = 0

    fold, call, bet = get_action_values(valid_actions, action)
    pot = round_state["pot"]["main"][AMOUNT]
    call_amount = call[AMOUNT]
    if action == Actions.FOLD:
        if call_amount == 0:
            act = CALL
            amount = 0
            reward = wrong_choose_reward  # Check is available
        else:
            act = FOLD
            amount = 0
            reward = right_choose_reward
    elif action == Actions.CHECK:
        if call_amount == 0:
            act = CALL
            amount = 0
            reward = right_choose_reward  # Check is available
        else:
            act = FOLD
            amount = 0
            reward = wrong_choose_reward
    elif action == Actions.CALL:
        if call_amount > 0:
            act = CALL
            amount = call[AMOUNT]
            reward = right_choose_reward
        else:
            act = FOLD
            amount = 0
            reward = wrong_choose_reward  # Call amount == 0
    elif action in (Actions.BET_25, Actions.BET_33, Actions.BET_50, Actions.BET_80, Actions.BET_150):
        if bet[AMOUNT][MIN_AMOUNT] == bet[AMOUNT][MAX_AMOUNT] == -1:  # Only call All_In
            amount = call_amount
            act = CALL
        elif r_only is False:
            mult = int(str(action).split('_')[1]) / 100
            b_a = call_amount + mult * pot
            if bet[AMOUNT][MIN_AMOUNT] <= b_a <= bet[AMOUNT][MAX_AMOUNT]:
                amount = b_a
            elif bet[AMOUNT][MIN_AMOUNT] > b_a:
                amount = bet[AMOUNT][MIN_AMOUNT]  # ALL_IN chosen
            else:
                amount = bet[AMOUNT][MAX_AMOUNT]  # ALL_IN chosen
            act = RAISE
            reward = right_choose_reward
        else:
            act = RAISE
            amount = bet[AMOUNT][MAX_AMOUNT]  # ALL_IN chosen because need to raise
            reward = wrong_choose_reward  # Need to predict BET_xxx answer
        reward += aggressive_bonus
    elif action in (Actions.RAISE_3BET, Actions.RAISE_3POT):
        if bet[AMOUNT][MIN_AMOUNT] == bet[AMOUNT][MAX_AMOUNT] == -1:  # Only call All_In
            amount = call_amount
            act = CALL
        elif r_only is True:
            b_a = call_amount + 3 * call_amount if action == Actions.RAISE_3BET else call_amount + 3 * pot
            if bet[AMOUNT][MIN_AMOUNT] <= b_a <= bet[AMOUNT][MAX_AMOUNT]:
                amount = b_a
            elif bet[AMOUNT][MIN_AMOUNT] > b_a:
                amount = bet[AMOUNT][MIN_AMOUNT]  # ALL_IN chosen
            else:
                amount = bet[AMOUNT][MAX_AMOUNT]  # ALL_IN chosen
            act = RAISE
            reward = right_choose_reward
        else:
            act = RAISE
            amount = bet[AMOUNT][MAX_AMOUNT]  # ALL_IN chosen because need to raise
            reward = wrong_choose_reward  # Need to predict RAISE_xxx answer
        reward += aggressive_bonus
    elif action == Actions.ALL_IN:
        if bet[AMOUNT][MIN_AMOUNT] == bet[AMOUNT][MAX_AMOUNT] == -1:
            amount = call_amount
            act = CALL
        else:
            act = RAISE
            amount = bet[AMOUNT][MAX_AMOUNT]
        reward = right_choose_reward + aggressive_bonus
    return act, int(amount), reward
