import napiod
import numpy as np
import matplotlib.pyplot as plt


def main():

    # Randomly generate the parameters for the order book
    phis = napiod.model.generate_random_transition_probabilities(
        with_agent=False)
    nus, alphas, betas = napiod.model.generate_random_hawkes_parameters(
        with_agent=False)

    # Define the parameters of the agent
    agent_base_rate = .25
    agent_self_excitation = np.zeros(
        (napiod.model.TOTAL_NUMBER_OF_EVENT_TYPES,
         napiod.model.TOTAL_NUMBER_OF_STATES),
        dtype=float)
    agent_self_excitation[
        :,
        list(
            napiod.model.state_indexes_with_negative_imbalances())] = 1.
    agent_self_decay = 2.0 * np.ones(
        (napiod.model.TOTAL_NUMBER_OF_EVENT_TYPES,
         napiod.model.TOTAL_NUMBER_OF_STATES),
        dtype=float)
    agent_impact_on_others = np.zeros(
        (napiod.model.TOTAL_NUMBER_OF_STATES,
         napiod.model.NUMBER_OF_ORDERBOOK_EVENTS),
        dtype=float
    )
    agent_impact_on_others[:, 0] = .1
    agent_impact_on_others[:, 1] = .5
    agent_impact_on_others[:, 3] = .15
    agent_decay_on_others = 0.85 * np.ones(
        (napiod.model.TOTAL_NUMBER_OF_STATES,
         napiod.model.NUMBER_OF_ORDERBOOK_EVENTS),
        dtype=float
    )
    agent_state_transition = np.zeros(
        (napiod.model.TOTAL_NUMBER_OF_STATES,
         napiod.model.TOTAL_NUMBER_OF_STATES),
        dtype=float
    )
    agent_state_transition[:, [0, 1, 2,  3]] = .25

    # Initialise the price impact model
    direction = -1
    price_impact = napiod.model.PriceImpact(direction)
    # Set the parameters of the order book and of the agent
    price_impact.set_orderbook_and_agent(
        phis,
        nus,
        alphas,
        betas,
        agent_state_transition,
        agent_base_rate,
        agent_impact_on_others,
        agent_self_excitation,
        agent_decay_on_others,
        agent_self_decay,
    )

    # Simulate the dynamic evolution of the order book while the agent executes their order
    execution_start = 0.
    execution_end = 100.
    horizon = 200.
    initial_state = 3
    times, events, states = price_impact.simulate_execution(
        execution_start,
        execution_end,
        horizon,
        initial_state=initial_state
    )

    # Reconstruct the price path from the simulated states of the model
    price = napiod.model.price_path_from_states(0, states)  # NOQA

    # Compute the agent price impact profile
    # Note: this assumes execution_start = 0.
    impact_profile = price_impact.compute_price_impact_profile(
        times, events, states,
        horizon,
        execution_end,
        dt=.5,
    )

    # Visualisations
    plot_time_start = 95.
    plot_time_end = 105.

    # Plot the price path from the simulated states of the model
    fig_price, _ = napiod.plot.price_path(
        times,
        events,
        states,
        plot_time_start,
        plot_time_end,
        initial_price=0
    )

    # Plot the evolution of the order book as encoded in the state-dependent Hawkes process
    fig_sample, _ = napiod.plot.hawkes_sample(
        times,
        events,
        states,
        price_impact.orderbook,
        plot_time_start,
        plot_time_end,
    )

    # Plot the agent's price impact profile
    fig_impact, _ = napiod.plot.price_impact_profile(
        impact_profile,
        plot_time_start,
        plot_time_end
    )

    return fig_price, fig_sample, fig_impact


if __name__ == '__main__':
    fig_price, fig_sample, fig_impact = main()
    plt.show()
