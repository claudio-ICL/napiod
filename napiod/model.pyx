from typing import Sequence, Optional
import numpy as np
cimport numpy as np
import scipy

from napiod import impact
from mpoints import hybrid_hawkes_exp

DTYPEf = np.float64
DTYPEi = np.int64
DTYPEb = np.uint8
ctypedef np.float64_t DTYPEf_t
ctypedef np.int64_t DTYPEi_t
ctypedef np.uint8_t DTYPEb_t

EVENTS = (
        'agent',  # Agent executes a trade by sending a market order
        'sell_mo', # Sell market order, i.e. execution of order on the bid side 
        'buy_mo', # Buy market order, i.e. execution of order on the ask side 
        'deflo', # Deflationary limit order, i.e. a new ask limit order is posted with price limit lower than the existing best ask price, or the existing best bid limit order is cancelled 
        'inflo', # Inflationary limit order, i.e. a new bid limit order is posted with price limit higher than the existing best bid price, or the existing best ask limit order is cancelled
        )

STATES = (
        'd-', # Down-negative, i.e. mid-price drops and volume imbalance is negative ( <  - .33)
        'd0', # Down-neutral, i.e. mid-price drops and volume imbalance is neutral (> - 0.33, < + 0.33)
        'd+', # Down-positive, i.e. mid-price drops and volume imbalance is positive (> + 0.33)
        's-', # Stable-negative, i.e. mid-price is unchanged and volume imbalance is negative ( <  - .33)
        's0', # Stable-neutral, i.e. mid-price is unchanged and volume imbalance is neutral ( >  - .33, < +.33)
        's+', # Stable-positive, i.e. mid-price is unchanged and volume imbalance is positive (< +.33)
        'u-', # Up-negative, i.e. mid-price increases and volume imbalance is negative ( <  - .33)
        'u0', # Up-neutral, i.e. mid-price increases and volume imbalance is neutral ( >  - .33, < +.33)
        'u+', # Up-positive, i.e. mid-price increases and volume imbalance is positive (< +.33)
)

AGENT_INDEX = 0  
NUMBER_OF_ORDERBOOK_EVENTS = 4
TOTAL_NUMBER_OF_EVENT_TYPES = 1 + NUMBER_OF_ORDERBOOK_EVENTS
NUMBER_OF_POSSIBLE_PRICE_MOVEMENTS = 3
NUMBER_OF_IMBALANCE_SEGMENTS = 3
TOTAL_NUMBER_OF_STATES = NUMBER_OF_POSSIBLE_PRICE_MOVEMENTS * NUMBER_OF_IMBALANCE_SEGMENTS

class Constants:
    @staticmethod
    def non_inflationary_events():
        return [1, 3]

    @staticmethod
    def non_deflationary_events():
        return [2, 4]

    @staticmethod
    def stationary_states():
        return list(stationary_state_indexes())

    @staticmethod
    def non_inflationary_states():
        return sorted(
            list(deflationary_state_indexes()) +
            list(stationary_state_indexes())
        )

    @staticmethod
    def non_deflationary_states():
        return sorted(
            list(inflationary_state_indexes()) +
            list(stationary_state_indexes())
        )

    @staticmethod
    def shape_of_transition_matrix():
        return (
            TOTAL_NUMBER_OF_STATES,
            TOTAL_NUMBER_OF_EVENT_TYPES,
            TOTAL_NUMBER_OF_STATES,
        )

    @staticmethod
    def shape_of_impact_coefficients():
        return (
            TOTAL_NUMBER_OF_EVENT_TYPES,
            TOTAL_NUMBER_OF_STATES,
            TOTAL_NUMBER_OF_EVENT_TYPES,
        )

    @staticmethod
    def shape_of_decay_coefficients():
        return (
            TOTAL_NUMBER_OF_EVENT_TYPES,
            TOTAL_NUMBER_OF_STATES,
            TOTAL_NUMBER_OF_EVENT_TYPES,
        )


class PriceImpact:

    def __init__(self, int direction):
        assert direction != 0
        self.direction = direction
        self.orderbook = hybrid_hawkes_exp.HybridHawkesExp(
                TOTAL_NUMBER_OF_EVENT_TYPES,
                TOTAL_NUMBER_OF_STATES,
                list(EVENTS),
                list(STATES),
                )

    def compute_price_impact_profile(
            self,
            np.ndarray[DTYPEf_t, ndim=1] times,
            np.ndarray[DTYPEi_t, ndim=1] events,
            np.ndarray[DTYPEi_t, ndim=1] states,
            horizon = None,
            execution_end = None,
            dt = None,
            ):
        if dt is None:
            dt = .25 * times[0]
        if horizon is None:
            horizon = times[-1] + 40 * dt
        if execution_end is None:
            execution_end = times[-1]
        profile = impact.price_impact_profile(
                self.direction,
                horizon,
                execution_end,
                dt,
                self.orderbook.transition_probabilities,
                self.orderbook.base_rates,
                self.orderbook.impact_coefficients,
                self.orderbook.decay_coefficients,
                times,
                events,
                states,
                )
        return profile

    def simulate_execution(
            self,
            execution_start = None,
            execution_end = None,
            horizon = None,
            initial_condition_times: Optional[Sequence] = None,
            initial_condition_events: Optional[Sequence] = None,
            initial_condition_states: Optional[Sequence] = None,
            initial_partial_sums: Optional[np.array] = None,
            initial_state: int = 0,
            ):
        if horizon is None:
            if execution_end is None:
                raise ValueError(
                        'You need to specify at least one of `execution_end` and `horizon`')
            else:
                horizon = 1.05 * execution_end
        if execution_end is None:
            execution_end = horizon
        if execution_start is None:
            execution_start = 0.
        assert execution_start < execution_end
        assert execution_end <= horizon
        # Instantiate `simulator` of the class `hybrid_hawkes_exp.HybridHawkesExp`
        # and initialite it with the parameters from `orderbook`.
        simulator = hybrid_hawkes_exp.HybridHawkesExp(
                TOTAL_NUMBER_OF_EVENT_TYPES,
                TOTAL_NUMBER_OF_STATES,
                list(EVENTS),
                list(STATES),
                )
        simulator.set_transition_probabilities(
                self.orderbook.transition_probabilities)
        simulator.set_hawkes_parameters(
                self.orderbook.base_rates,
                self.orderbook.impact_coefficients,
                self.orderbook.decay_coefficients
                )
        # Simulate dynamics up to the end of the execution
        times, events, states = simulator.simulate(
                execution_start,
                execution_end,
                initial_condition_times,
                initial_condition_events,
                initial_condition_states,
                initial_partial_sums,
                )
        # Save the status of Hawkes kernel at the end of the execution
        ps = simulator.compute_partial_sums(
            times,
            events,
            states,
            execution_end,
            initial_partial_sums=initial_partial_sums,
            time_initial_condition=execution_start
            )
        # Set the agent's parameters to zero, as the execution has ended
        phis = simulator.transition_probabilities[:, 1:, :]
        nus = simulator.base_rates[1:]
        alphas = simulator.impact_coefficients[1:, :, 1:]
        betas = simulator.decay_coefficients[1:, :, 1:]
        naphis, nanus, naalphas, nabetas = _add_agent_params(
                self.direction,
                phis,
                nus,
                alphas,
                betas,
                )
        simulator.set_transition_probabilities(naphis)
        simulator.set_hawkes_parameters(
                nanus, naalphas, nabetas)
        # simulate the order book after the execution, up to the horizon
        times, events, states = simulator.simulate(
                execution_end,
                horizon,
                times,
                events,
                states,
                ps,
                states[-1],
                )
        # Returns times, events and states
        return times, events, states


    def set_orderbook_and_agent(
            self,
            transition_probabilities,
            nus,
            alphas,
            betas,
            agent_state_transition=None,
            agent_base_rate=None,
            agent_impact_on_others=None,
            agent_self_excitation=None,
            agent_decay_on_others=None,
            agent_self_decay=None,
        ):
        phis, nus, alphas, betas = _add_agent_params(
                self.direction,
                transition_probabilities,
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
        _check_transition_probabilities(phis, with_agent=1, agent_direction=self.direction)
        self.orderbook.set_transition_probabilities(phis)
        self.orderbook.set_hawkes_parameters(nus, alphas, betas)
                


def idem_event(int direction):
    cdef Py_ssize_t e = 0 if direction < 0 else 1
    return e


def generate_random_hawkes_parameters(
        int with_agent = 1,
        ):
    dx = TOTAL_NUMBER_OF_STATES
    if with_agent:
        de = TOTAL_NUMBER_OF_EVENT_TYPES
    else:
        de = NUMBER_OF_ORDERBOOK_EVENTS
    nus = np.random.uniform(low=0., high=1., size=(de,))
    alphas = .5 * \
        scipy.sparse.random(de, dx*de, density=.50).A.reshape(de, dx, de)
    betas = np.random.uniform(low=.95, high=1.05, size=(de, dx, de))
    return nus, alphas, betas


def generate_random_transition_probabilities(
        int with_agent = 1,
        int agent_direction = -1,
        ):
    if with_agent:
        inflationary_events = [4]
        non_inflationary_events = [1, 3]
        deflationary_events = [3]
        non_deflationary_events = [2, 4]
        shape = (
                    TOTAL_NUMBER_OF_STATES, 
                    TOTAL_NUMBER_OF_EVENT_TYPES, 
                    TOTAL_NUMBER_OF_STATES)
    else:
        inflationary_events = [3]
        non_inflationary_events = [0, 2]
        deflationary_events = [2]
        non_deflationary_events = [1, 3]
        shape = (
                    TOTAL_NUMBER_OF_STATES, 
                    NUMBER_OF_ORDERBOOK_EVENTS, 
                    TOTAL_NUMBER_OF_STATES)
    stationary_states = list(stationary_state_indexes())
    non_inflationary_states = sorted(
            list(deflationary_state_indexes()) + 
            list(stationary_state_indexes())
    )
    non_deflationary_states = sorted(
            list(inflationary_state_indexes()) + 
            list(stationary_state_indexes())
    )
    cdef np.ndarray[DTYPEf_t, ndim=3] phis = np.zeros(shape, dtype=DTYPEf)
    for e in non_inflationary_events:
        phis[:, e, non_inflationary_states] += scipy.sparse.random(
                TOTAL_NUMBER_OF_STATES,
                len(non_inflationary_states),
                density=.95
                )
        if e in deflationary_events:
            phis[:, e, stationary_states] = 0.
        phis[:, e, :] /= np.sum(phis[:, e, :], axis=1, keepdims=True)
    for e in non_deflationary_events:
        phis[:, e, non_deflationary_states] += scipy.sparse.random(
                TOTAL_NUMBER_OF_STATES,
                len(non_deflationary_states),
                density=.95
                )
        if e in inflationary_events:
            phis[:, e, stationary_states] = 0.
        phis[:, e, :] /= np.sum(phis[:, e, :], axis=1, keepdims=True)
    if with_agent:
        if agent_direction < 0:
            after_agent_states = non_inflationary_states
        else:
            after_agent_states = non_deflationary_states
        phis[:, AGENT_INDEX, after_agent_states] = np.random.uniform(
                low=0., 
                high=1., 
                size=(TOTAL_NUMBER_OF_STATES, len(after_agent_states))
        )
    cdef np.ndarray[DTYPEf_t, ndim=3] masses = np.sum(phis, axis=2, keepdims=True)
    phis /= masses
    _check_transition_probabilities(phis, with_agent, agent_direction)
    return phis


def _check_transition_probabilities(
        np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
        int with_agent,
        agent_direction=None,
        ):
    if with_agent:
        assert (
                transition_probabilities.shape[0], 
                transition_probabilities.shape[1], 
                transition_probabilities.shape[2]
                ) == (
                        TOTAL_NUMBER_OF_STATES, 
                        TOTAL_NUMBER_OF_EVENT_TYPES, 
                        TOTAL_NUMBER_OF_STATES)
    else:
        assert (
                transition_probabilities.shape[0], 
                transition_probabilities.shape[1], 
                transition_probabilities.shape[2]
                ) == (
                        TOTAL_NUMBER_OF_STATES, 
                        NUMBER_OF_ORDERBOOK_EVENTS, 
                        TOTAL_NUMBER_OF_STATES)
    masses = np.sum(transition_probabilities, axis=2)
    expected_masses = np.ones_like(masses)
    assert np.allclose(
            masses,
            expected_masses,
            rtol=1e-8,
            atol=1e-10,
            )
    if with_agent:
        inflationary_events = [4]
        non_inflationary_events = [1, 3]
        deflationary_events = [3]
        non_deflationary_events = [2, 4]
    else:
        inflationary_events = [3]
        non_inflationary_events = [0, 2]
        deflationary_events = [2]
        non_deflationary_events = [1, 3]
    if with_agent:
        if not (agent_direction is None):
            if agent_direction > 0:
                non_deflationary_events += [AGENT_INDEX]
            else:
                non_inflationary_events += [AGENT_INDEX]
    for e in non_inflationary_events:
        assert np.allclose(
                transition_probabilities[:, e, list(inflationary_state_indexes())],
                0.,
                )
    for e in non_deflationary_events:
        assert np.allclose(
                transition_probabilities[:, e, list(deflationary_state_indexes())],
                0.,
                )
    for e in inflationary_events + deflationary_events:
        assert np.allclose(
                transition_probabilities[:, e, list(stationary_state_indexes())],
                0.,
                )

def _add_agent_params(
        int direction,
        np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
        np.ndarray[DTYPEf_t, ndim=1] nus,
        np.ndarray[DTYPEf_t, ndim=3] alphas, 
        np.ndarray[DTYPEf_t, ndim=3] betas, 
        np.ndarray[DTYPEf_t, ndim=2] agent_state_transition = None,
        agent_base_rate = None,
        np.ndarray[DTYPEf_t, ndim=2] agent_impact_on_others = None,
        np.ndarray[DTYPEf_t, ndim=2] agent_self_excitation = None,
        np.ndarray[DTYPEf_t, ndim=2] agent_decay_on_others = None,
        np.ndarray[DTYPEf_t, ndim=2] agent_self_decay = None,
        ):
    # if `agent_base_rate` and `agent_self_excitation` are not given,
    # they are set to zero by default, resulting in an inactive agent
    assert nus.shape[0] == NUMBER_OF_ORDERBOOK_EVENTS
    assert (alphas.shape[0], alphas.shape[1], alphas.shape[2]) == (
            NUMBER_OF_ORDERBOOK_EVENTS,
            TOTAL_NUMBER_OF_STATES,
            NUMBER_OF_ORDERBOOK_EVENTS,
            )
    assert (betas.shape[0], betas.shape[1], betas.shape[2]) == (
            NUMBER_OF_ORDERBOOK_EVENTS,
            TOTAL_NUMBER_OF_STATES,
            NUMBER_OF_ORDERBOOK_EVENTS,
            )
    cdef Py_ssize_t e = idem_event(direction)
    _check_transition_probabilities(transition_probabilities, with_agent=0)
    if agent_state_transition is None:
        agent_state_transition = np.array(
                transition_probabilities[:, e, :],
                copy=True,
                )
    assert (agent_state_transition.shape[0], agent_state_transition.shape[1]) == (
            TOTAL_NUMBER_OF_STATES,
            TOTAL_NUMBER_OF_STATES,
            )
    assert np.allclose(
            np.sum(agent_state_transition, axis=1),
            1.,
            rtol=1e-6,
            atol=1e-10,
            )
    if agent_base_rate is None:
        agent_base_rate = 0.
    if agent_impact_on_others is None:
        agent_impact_on_others = np.array(
                alphas[e, :, :],
                copy=True,
                )
    assert (agent_impact_on_others.shape[0], agent_impact_on_others.shape[1]) == (
            TOTAL_NUMBER_OF_STATES,
            NUMBER_OF_ORDERBOOK_EVENTS,
            )
    if agent_self_excitation is None:
        agent_self_excitation = np.zeros(
                (TOTAL_NUMBER_OF_EVENT_TYPES, TOTAL_NUMBER_OF_STATES), 
                dtype=DTYPEf)
    assert (agent_self_excitation.shape[0], agent_self_excitation.shape[1]) == (
            TOTAL_NUMBER_OF_EVENT_TYPES, TOTAL_NUMBER_OF_STATES)
    if agent_decay_on_others is None:
        agent_decay_on_others = np.array(
                betas[e, :, :],
                copy=True,
                )
    assert (agent_decay_on_others.shape[0], agent_decay_on_others.shape[1]) == (
            TOTAL_NUMBER_OF_STATES,
            NUMBER_OF_ORDERBOOK_EVENTS,
            )
    if agent_self_decay is None:
        agent_self_decay = np.ones(
                (TOTAL_NUMBER_OF_EVENT_TYPES, TOTAL_NUMBER_OF_STATES), 
                dtype=DTYPEf)
    assert (agent_self_decay.shape[0], agent_self_decay.shape[1]) == (
            TOTAL_NUMBER_OF_EVENT_TYPES, TOTAL_NUMBER_OF_STATES)
    
    # Expand transition probabilities
    transition_probabilities = np.insert(
            transition_probabilities,
            obj=0,
            values=agent_state_transition,
            axis=1,
            )

    # Expand the base rates
    nus = np.insert(
            nus,
            obj=0,
            values=agent_base_rate,
            axis=0
            )

    # Expand the given alphas and betas to account for the effect of the agents on existing events
    alphas = np.insert(alphas,
                       obj=0,
                       values=agent_impact_on_others,
                       axis=0)
    betas = np.insert(betas,
                       obj=0,
                       values=agent_decay_on_others,
                       axis=0)
    # Add to the alphas and betas the coeffieicnt that account for the self excitation of the agent's events
    alphas = np.insert(
            alphas,
            obj=0,
            values=agent_self_excitation,
            axis=2)
    betas = np.insert(
            betas,
            obj=0,
            values=agent_self_decay,
            axis=2)

    return transition_probabilities, nus, alphas, betas





def state_variable_to_state_index(
        int price_change,
        int imbalance,
        int num_discretisations = NUMBER_OF_IMBALANCE_SEGMENTS,
        ):
    """
    `price_change` takes values -1, 0, +1, indicating price decrease, price unchanged, price increase.
    Assuming `num_discretisations` is odd, `imbalance` takes values from -(num_discretisations - 1) / 2 to +num_discretisations - 1)/2. 

    """
    return (price_change + 1) * num_discretisations + (num_discretisations // 2 + imbalance)

def state_indexes_with_positive_imbalances():
    for price_change in (-1, 0, 1):
        yield state_variable_to_state_index(price_change, 1)

def state_indexes_with_negative_imbalances():
    for price_change in (-1, 0, 1):
        yield state_variable_to_state_index(price_change, -1)

def price_changes_from_states(
        np.ndarray[DTYPEi_t, ndim=1] states
        ):
    cdef np.ndarray[DTYPEi_t, ndim=1] price_changes = -1 + (states // NUMBER_OF_IMBALANCE_SEGMENTS)
    return price_changes

def price_path_from_states(
        long initial_price_in_ticks,
        np.ndarray[DTYPEi_t, ndim=1] states
        ):
    cdef np.ndarray[DTYPEi_t, ndim=1] price_changes = price_changes_from_states(states)
    cdef np.ndarray[DTYPEi_t, ndim=1] price_path = initial_price_in_ticks + np.cumsum(price_changes)
    return price_path

def price_change_from_state_index(
        int x,
        int num_discretisations = NUMBER_OF_IMBALANCE_SEGMENTS,
        ):
    cdef int price_change = -1 + (x // num_discretisations)
    return price_change

def imbalance_from_state_index(
        int x,
        int num_discretisations = NUMBER_OF_IMBALANCE_SEGMENTS,
        ):
    cdef int imbalance = (x %  num_discretisations) - (num_discretisations // 2)
    return imbalance

def imbalances_from_states(
        np.ndarray[DTYPEi_t, ndim=1] states,
        int num_discretisations = NUMBER_OF_IMBALANCE_SEGMENTS,
        ):
    cdef np.ndarray[DTYPEi_t, ndim=1] imbalances = (states % num_discretisations) - (num_discretisations // 2)
    return imbalances

def stationary_state_indexes(
        int num_discretisations = NUMBER_OF_IMBALANCE_SEGMENTS,
        ):
    for imbalance in range(-(num_discretisations //2), 1 + (num_discretisations // 2)):
        yield state_variable_to_state_index(0, imbalance, num_discretisations)

def _stationary_state_indexes_recon(
        int num_discretisations = NUMBER_OF_IMBALANCE_SEGMENTS,
        ):
    for x in range(1*num_discretisations, 2*num_discretisations):
        yield x


def inflationary_state_indexes(
        int num_discretisations = NUMBER_OF_IMBALANCE_SEGMENTS,
        ):
    for imbalance in range(-(num_discretisations //2), 1 + (num_discretisations // 2)):
        yield state_variable_to_state_index(1, imbalance, num_discretisations)

def _inflationary_state_indexes_recon(
        int num_discretisations = NUMBER_OF_IMBALANCE_SEGMENTS,
        ):
    for x in range(2*num_discretisations, 3*num_discretisations):
        yield x


def deflationary_state_indexes(
        int num_discretisations = NUMBER_OF_IMBALANCE_SEGMENTS,
        ):
    for imbalance in range(-(num_discretisations //2), 1+ (num_discretisations // 2)):
        yield state_variable_to_state_index(-1, imbalance, num_discretisations)


def _deflationary_state_indexes_recon(
        int num_discretisations = NUMBER_OF_IMBALANCE_SEGMENTS,
        ):
    for x in range(num_discretisations):
        yield x

        
