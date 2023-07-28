import numpy as np
import napiod


stationary_states = list(napiod.model.stationary_state_indexes())
non_inflationary_states = sorted(
    list(napiod.model.deflationary_state_indexes()) +
    list(napiod.model.stationary_state_indexes())
)
non_deflationary_states = sorted(
    list(napiod.model.inflationary_state_indexes()) +
    list(napiod.model.stationary_state_indexes())
)
shape_of_transition_matrix = (
    napiod.model.TOTAL_NUMBER_OF_STATES,
    napiod.model.TOTAL_NUMBER_OF_EVENT_TYPES,
    napiod.model.TOTAL_NUMBER_OF_STATES,
)

agent_direction = -1  # The agents is selling


def balanced_order_book():
    # low volatility
    pass


def balanced_order_book_with_herding():
    # high volatility
    pass


def bullish_unbalanced_order_book():
    pass


def trend_following_agent():
    pass


def contrarian_agent():
    pass


def market_provides_liquidity():
    pass


def market_withdraws_liquidity():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
