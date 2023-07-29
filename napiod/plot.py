from napiod import model
import numpy as np
import copy
import bisect
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpoints.plot_tools import (
    sample_path as hawkes_sample,
    kernels_exp as norms_of_hawkes_kernels,
)


def price_path(
        times,
        events,
        states,
        time_start,
        time_end,
        initial_price=0,
        num_subplots=2,
):
    # Reconstruct the price path from the states of the model
    prices = model.price_path_from_states(initial_price, states)
    # Reconstruct the eveolution of volume imbalances from the states of the model
    imbalances = model.imbalances_from_states(states)
    t0 = bisect.bisect_left(times, time_start)
    t1 = bisect.bisect_right(times, time_end)
    prices = list(copy.copy(prices[t0: t1]))
    imbalances = list(copy.copy(imbalances[t0: t1]))
    times = list(copy.copy(times[t0: t1]))

    f, axes = plt.subplots(num_subplots, 1, sharex='col')

    # Price path
    ax_price = axes[0]
    ax_price.step(times, prices, where='post', linewidth=2, label='price')
    ax_price.set_ylabel('Tick')
    ax_price.set_xlabel('Time')
    ax_price.legend()

    # Imabalance path
    ax_imb = axes[1]
    ax_imb.step(times, imbalances, where='post',
                linewidth=2, label='volume imbalance', color='darkgreen')
    ax_imb.set_xlabel('Time')
    ax_imb.set_ylabel('Volume imbalance')
    ax_imb.set_yticks([-1, 0, 1])
    ax_imb.set_yticklabels(['negative', 'neutral', 'positive'])

    plt.tight_layout()
    ax_imb.legend()
    return f, axes


def price_impact_profile(
        profile,
        time_start,
        time_end,
):
    t0 = bisect.bisect(profile[:, 0], time_start)
    t1 = bisect.bisect(profile[:, 0], time_end)

    f, axes = plt.subplots(3, 1, sharex='col')

    ax_tot = axes[0]
    ax_tot.plot(
        profile[t0:t1, 0],
        profile[t0:t1, 3],
        linewidth=2,
        color='blue',
        label='total impact'
    )
    ax_tot.legend()

    ax_direct = axes[1]
    ax_direct.plot(
        profile[t0:t1, 0],
        profile[t0:t1, 1],
        linewidth=2,
        color='darkorange',
        label='direct impact'
    )
    ax_direct.legend()

    ax_indirect = axes[2]
    ax_indirect.plot(
        profile[t0:t1, 0],
        profile[t0:t1, 2],
        linewidth=2,
        color='darkgreen',
        label='indirect impact'
    )
    ax_indirect.legend()
    ax_indirect.set_xlabel('Time')

    return f, axes


def agent_execution(
        times,
        events,
        time_start,
        time_end,
        fig=None,
):
    is_agent = events == model.AGENT_INDEX
    agent_times = np.array(
        times[is_agent],
        copy=True)
    agent_counter = 1 + np.arange(np.sum(is_agent))
    t0 = bisect.bisect(agent_times, time_start)
    t1 = bisect.bisect_right(agent_times, time_end)
    pret0 = max(0, t0 - 1)
    pret1 = max(pret0, t1 - 1)
    plot_times = np.concatenate(
        ([time_start], agent_times[t0: t1], [time_end]))
    plot_counter = np.concatenate(
        (agent_counter[[pret0]], agent_counter[t0: t1], agent_counter[[pret1]]))

    if fig is None:
        fig, axes = plt.subplots(1, 1)
        ax = axes
    else:
        ax = fig.axes[-1]

    ax.step(plot_times, plot_counter, where='post', color='red', label="agent")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of market orders")
    ax.legend()

    return fig, fig.axes


def price_path_and_agent_execution(
        times,
        events,
        states,
        time_start,
        time_end,
        initial_price=0,
):
    f, axes = price_path(
        times,
        events,
        states,
        time_start,
        time_end,
        initial_price,
        3
    )
    f, axes = agent_execution(
        times,
        events,
        time_start,
        time_end,
        fig=f
    )
    return f, axes
