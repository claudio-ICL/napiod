import unittest
import numpy as np
import scipy
import bisect

from mpoints import hybrid_hawkes_exp
from napiod import model


class ModelTest(unittest.TestCase):
    def setUp(self):
        number_of_event_types: int = model.NUMBER_OF_ORDERBOOK_EVENTS
        de = number_of_event_types
        number_of_states: int = model.TOTAL_NUMBER_OF_STATES
        dx = number_of_states
        events_labels = [
            'sell_mo',
            'buy_mo',
            'deflo',
            'inflo',
        ]
        states_labels = [
            'd_neg',
            'd_neu',
            'd_pos',
            'f_neg',
            'f_neu',
            'f_pos',
            'u_neg',
            'u_neu',
            'u_pos',
        ]
        phis = model.generate_random_transition_probabilities(with_agent=False)
        nus = np.random.uniform(low=0., high=1., size=(de,))
        alphas = .5 * \
            scipy.sparse.random(de, dx*de, density=.50).A.reshape(de, dx, de)
        betas = np.random.uniform(low=.95, high=1.05, size=(de, dx, de))
        # `hhe` represents the order book without the agent
        hhe = hybrid_hawkes_exp.HybridHawkesExp(
            number_of_event_types,
            number_of_states,
            events_labels,
            states_labels,
        )
        hhe.set_transition_probabilities(phis)
        hhe.set_hawkes_parameters(nus, alphas, betas)
        self.hhe = hhe

    def test_labels(self):
        self.assertEqual(
            len(model.EVENTS),
            model.TOTAL_NUMBER_OF_EVENT_TYPES,
        )
        self.assertEqual(
            len(model.STATES),
            model.TOTAL_NUMBER_OF_STATES,
        )

    def test_add_seller_agent(self):
        direction = -1
        self._test_add_agent_params(direction)

    def test_add_buyer_agent(self):
        direction = 1
        self._test_add_agent_params(direction)

    def _test_add_agent_params(self, direction):
        idem_event = model.idem_event(direction)

        def _test(agent_base_rate=None):
            # Expand the parameters of the Hawkes process `hhe` in order to include the agent.
            phis, nus, alphas, betas = model._add_agent_params(
                direction,
                self.hhe.transition_probabilities,
                self.hhe.base_rates,
                self.hhe.impact_coefficients,
                self.hhe.decay_coefficients,
                agent_base_rate=agent_base_rate,
            )
            self.assertEqual(
                phis.shape,
                (model.TOTAL_NUMBER_OF_STATES,
                 model.TOTAL_NUMBER_OF_EVENT_TYPES,
                 model.TOTAL_NUMBER_OF_STATES),
            )
            self.assertEqual(
                nus.shape,
                (model.TOTAL_NUMBER_OF_EVENT_TYPES,),
            )
            self.assertEqual(
                alphas.shape,
                (model.TOTAL_NUMBER_OF_EVENT_TYPES,
                 model.TOTAL_NUMBER_OF_STATES,
                 model.TOTAL_NUMBER_OF_EVENT_TYPES
                 ),
            )
            self.assertEqual(
                betas.shape,
                (model.TOTAL_NUMBER_OF_EVENT_TYPES,
                 model.TOTAL_NUMBER_OF_STATES,
                 model.TOTAL_NUMBER_OF_EVENT_TYPES
                 ),
            )
            expected_base_rate = 0. if agent_base_rate is None else agent_base_rate
            self.assertTrue(
                np.allclose(
                    nus[0],
                    expected_base_rate,
                )
            )
            self.assertTrue(
                np.allclose(
                    alphas[0, :, 1:],
                    self.hhe.impact_coefficients[idem_event, :, :]
                )
            )
            self.assertTrue(
                np.allclose(
                    alphas[:, :, 0],
                    0.,
                )
            )
            self.assertTrue(
                np.allclose(
                    betas[0, :, 1:],
                    self.hhe.decay_coefficients[idem_event, :, :]
                )
            )

        # Test 1: we do not pass agent parameters, so they are all set to zero
        _test(None)

        # Test 2: Poisson execution
        _test(agent_base_rate=1.)

    def test_simulation_of_buyer_execution(self):
        direction = 1
        execution_start = 0.
        execution_end = 5000.
        horizon = 10000.
        self._test_simulation_of_execution(
            direction,
            execution_start,
            execution_end,
            horizon
        )

    def test_simulation_of_seller_execution(self):
        direction = -1
        execution_start = 0.
        execution_end = 5000.
        horizon = 10000.
        self._test_simulation_of_execution(
            direction,
            execution_start,
            execution_end,
            horizon
        )

    def _test_simulation_of_execution(
            self,
            direction,
            execution_start,
            execution_end,
            horizon,
    ):
        idem_event = model.idem_event(direction)
        # Poisson execution
        execution_base_rate = self.hhe.base_rates[idem_event]
        price_impact = model.PriceImpact(direction)
        price_impact.set_orderbook_and_agent(
            self.hhe.transition_probabilities,
            self.hhe.base_rates,
            self.hhe.impact_coefficients,
            self.hhe.decay_coefficients,
            agent_base_rate=execution_base_rate,
        )
        times, events, states = price_impact.simulate_execution(
            execution_start,
            execution_end,
            horizon,
            initial_state=1 + model.NUMBER_OF_IMBALANCE_SEGMENTS
        )
        self.assertTrue(
            np.all(np.diff(times) > 0.),
            'Event times are not increasing!',
        )
        self.assertTrue(
            np.all(
                events < model.TOTAL_NUMBER_OF_EVENT_TYPES
            ),
            'Unrecognised events in simulation'
        )
        self.assertTrue(
            np.all(
                states < model.TOTAL_NUMBER_OF_STATES
            ),
            'Unrecognised states in simulation'
        )
        t0 = bisect.bisect_left(times, execution_start)
        t1 = max(0, min(-1 + len(times), bisect.bisect_left(times, execution_end)))
        self.assertTrue(
            np.all(
                events[:t0] != model.AGENT_INDEX
            ),
            'Agent events found before execution starts',
        )
        self.assertTrue(
            np.all(
                events[t1:] != model.AGENT_INDEX
            ),
            'Agent events found after execution ends',
        )

        # Check states after inflationary events
        inflationary_event = 4
        is_inflationary_event = events == inflationary_event
        expected_states = list(model.inflationary_state_indexes())
        self.assertTrue(
            set(states[is_inflationary_event]).issubset(set(expected_states)),
        )

        # Check states after deflationary events
        deflationary_event = 3
        is_deflationary_event = events == deflationary_event
        expected_states = list(model.deflationary_state_indexes())
        self.assertTrue(
            set(states[is_deflationary_event]).issubset(set(expected_states)),
        )

        # Check states after agent's executions
        is_agent_event = events == model.AGENT_INDEX
        expected_states = list(model.stationary_state_indexes())
        if direction > 0:
            expected_states += list(model.inflationary_state_indexes())
        else:
            expected_states += list(model.deflationary_state_indexes())
        self.assertTrue(
            set(states[is_agent_event]).issubset(set(expected_states)),
        )

    def test_price_path_with_buyer(self):
        direction = 1
        execution_start = 0.
        execution_end = 5000
        horizon = 10000
        self._test_price_path(
            direction,
            execution_start,
            execution_end,
            horizon
        )

    def test_price_path_with_seller(self):
        direction = -1
        execution_start = 0.
        execution_end = 5000
        horizon = 10000
        self._test_price_path(
            direction,
            execution_start,
            execution_end,
            horizon
        )

    def _test_price_path(
            self,
            direction,
            execution_start,
            execution_end,
            horizon,
    ):
        idem_event = model.idem_event(direction)
        # Poisson execution
        execution_base_rate = self.hhe.base_rates[idem_event]
        price_impact = model.PriceImpact(direction)
        price_impact.set_orderbook_and_agent(
            self.hhe.transition_probabilities,
            self.hhe.base_rates,
            self.hhe.impact_coefficients,
            self.hhe.decay_coefficients,
            agent_base_rate=execution_base_rate,
        )
        times, events, states = price_impact.simulate_execution(
            execution_start,
            execution_end,
            horizon,
            initial_state=1 + model.NUMBER_OF_IMBALANCE_SEGMENTS
        )

        price_deltas = model.price_changes_from_states(states)
        self.assertEqual(
            set(price_deltas),
            {-1, 0, +1},
            'Unexpected price deltas',
        )

        inflationary_events = [4]
        deflationary_events = [3]
        non_inflationary_events = [1, 3]
        non_deflationary_events = [2, 4]
        if direction < 0:
            non_inflationary_events += [0]
        else:
            non_deflationary_events += [0]

        for e in inflationary_events:
            is_inflationary_event = events == e
            self.assertTrue(
                np.all(
                    price_deltas[is_inflationary_event] == 1
                ),
                f'price does not increase when inflationary event {e} occurs'
            )

        for e in deflationary_events:
            is_deflationary_event = events == e
            self.assertTrue(
                np.all(
                    price_deltas[is_deflationary_event] == -1
                ),
                f'price does not decrease when deflationary event {e} occurs'
            )

        for e in non_deflationary_events:
            is_nondelf_event = events == e
            self.assertTrue(
                np.all(
                    price_deltas[is_nondelf_event] != -1
                ),
                f'Price decreases when non deflationary event {e} occurs'
            )

        for e in non_inflationary_events:
            is_noninfl_event = events == e
            self.assertTrue(
                np.all(
                    price_deltas[is_noninfl_event] != 1
                ),
                f'Price increases when non inflationary event {e} occurs'
            )

    def test_state_conversions(self):
        for d in [3, 5, 7, 9]:
            for p in [-1, 0, +1]:
                for i in range(- (d // 2), 1 + d//2):
                    x = model.state_variable_to_state_index(
                        p, i, d)
                    p_ = model.price_change_from_state_index(
                        x, d)
                    i_ = model.imbalance_from_state_index(x, d)
                    self.assertEqual(
                        p, p_)
                    self.assertEqual(i, i_)
        self.assertEqual(
            list(model.stationary_state_indexes()),
            list(model._stationary_state_indexes_recon()),
            'Failed to reconcile stationary states',
        )
        self.assertEqual(
            list(model.inflationary_state_indexes()),
            list(model._inflationary_state_indexes_recon()),
            'Failed to reconcile inflationary states',
        )
        self.assertEqual(
            list(model.deflationary_state_indexes()),
            list(model._deflationary_state_indexes_recon()),
            'Failed to reconcile deflationary states',
        )

    def test_direct_impact_of_seller(self):
        direction = -1
        execution_end = 5000.
        horizon = 10000.
        self._test_direct_impact(
            direction,
            execution_end,
            horizon,
        )

    def test_direct_impact_of_buyer(self):
        direction = 1
        execution_end = 5000.
        horizon = 10000.
        self._test_direct_impact(
            direction,
            execution_end,
            horizon,
        )

    def _test_direct_impact(
            self,
            direction,
            execution_end,
            horizon,
    ):
        execution_start = 0.
        idem_event = model.idem_event(direction)
        # Poisson execution
        execution_base_rate = self.hhe.base_rates[idem_event]
        price_impact = model.PriceImpact(direction)
        price_impact.set_orderbook_and_agent(
            self.hhe.transition_probabilities,
            self.hhe.base_rates,
            self.hhe.impact_coefficients,
            self.hhe.decay_coefficients,
            agent_base_rate=execution_base_rate,
        )
        times, events, states = price_impact.simulate_execution(
            execution_start,
            execution_end,
            horizon,
            initial_state=1 + model.NUMBER_OF_IMBALANCE_SEGMENTS
        )
        profile = price_impact.compute_price_impact_profile(
            times,
            events,
            states,
            horizon,
            execution_end
        )
        self.assertTrue(
            np.all(
                profile[:, 1] >= 0.,
            ),
            'Direct impact should be non-negative',
        )
        self.assertTrue(
            np.all(
                np.diff(profile[:, 1]) >= 0.,
            ),
            'Direct impact should be non-decreasing',
        )
        t1 = bisect.bisect_right(profile[:, 0], execution_end)
        self.assertTrue(
            np.allclose(
                np.diff(profile[t1:, 1]),
                0.
            ),
            'Direct impact is not flat after execution has finished'
        )

    def test_indirect_impact_of_seller(self):
        direction = -1
        execution_end = 5000.
        horizon = 10000.
        self._test_indirect_impact(
            direction,
            execution_end,
            horizon
        )

    def test_indirect_impact_of_buyer(self):
        direction = 1
        execution_end = 5000.
        horizon = 10000.
        self._test_indirect_impact(
            direction,
            execution_end,
            horizon
        )

    def _test_indirect_impact(
            self,
            direction,
            execution_end,
            horizon
    ):
        execution_start = 0.
        idem_event = model.idem_event(direction)
        # Test 1: No impact of agent on other orderbook events
        execution_base_rate = self.hhe.base_rates[idem_event]
        agent_impact_on_others = np.zeros(
            (model.TOTAL_NUMBER_OF_STATES,
             model.NUMBER_OF_ORDERBOOK_EVENTS
             ),
            dtype=float)
        price_impact = model.PriceImpact(direction)
        price_impact.set_orderbook_and_agent(
            self.hhe.transition_probabilities,
            self.hhe.base_rates,
            self.hhe.impact_coefficients,
            self.hhe.decay_coefficients,
            agent_base_rate=execution_base_rate,
            agent_impact_on_others=agent_impact_on_others,
        )
        times, events, states = price_impact.simulate_execution(
            execution_start,
            execution_end,
            horizon,
            initial_state=1 + model.NUMBER_OF_IMBALANCE_SEGMENTS
        )
        profile = price_impact.compute_price_impact_profile(
            times,
            events,
            states,
            horizon,
            execution_end
        )
        self.assertTrue(
            np.allclose(
                profile[:, 2],
                0.,
                rtol=1e-7,
                atol=1e-12,
            ),
            'Indirect impact should be null when the agent has no impact on other orderbook events'
        )

        # Test 2: Only impact on events in the same direction
        same_events = [0, 2] if direction < 0 else [1, 3]
        agent_impact_on_others = np.zeros(
            (model.TOTAL_NUMBER_OF_STATES,
             model.NUMBER_OF_ORDERBOOK_EVENTS
             ),
            dtype=float)
        agent_impact_on_others[:, same_events] = np.random.uniform(
            low=0.,
            high=2.,
            size=(model.TOTAL_NUMBER_OF_STATES, 2)
        )
        price_impact = model.PriceImpact(direction)
        price_impact.set_orderbook_and_agent(
            self.hhe.transition_probabilities,
            self.hhe.base_rates,
            self.hhe.impact_coefficients,
            self.hhe.decay_coefficients,
            agent_base_rate=execution_base_rate,
            agent_impact_on_others=agent_impact_on_others,
        )
        times, events, states = price_impact.simulate_execution(
            execution_start,
            execution_end,
            horizon,
            initial_state=1 + model.NUMBER_OF_IMBALANCE_SEGMENTS
        )
        profile = price_impact.compute_price_impact_profile(
            times,
            events,
            states,
            horizon,
            execution_end
        )
        self.assertTrue(
            np.all(
                profile[:, 2] >= 0.,
            ),
            'Indirect price impact should be non-negative when agent has impact only on same events',
        )

        # Test 3: Only impact on events in the opposite direction
        opposite_events = [0, 2] if direction > 0 else [1, 3]
        agent_impact_on_others = np.zeros(
            (model.TOTAL_NUMBER_OF_STATES,
             model.NUMBER_OF_ORDERBOOK_EVENTS
             ),
            dtype=float)
        agent_impact_on_others[:, opposite_events] = np.random.uniform(
            low=0.,
            high=2.,
            size=(model.TOTAL_NUMBER_OF_STATES, 2)
        )
        price_impact = model.PriceImpact(direction)
        price_impact.set_orderbook_and_agent(
            self.hhe.transition_probabilities,
            self.hhe.base_rates,
            self.hhe.impact_coefficients,
            self.hhe.decay_coefficients,
            agent_base_rate=execution_base_rate,
            agent_impact_on_others=agent_impact_on_others,
        )
        times, events, states = price_impact.simulate_execution(
            execution_start,
            execution_end,
            horizon,
            initial_state=1 + model.NUMBER_OF_IMBALANCE_SEGMENTS
        )
        profile = price_impact.compute_price_impact_profile(
            times,
            events,
            states,
            horizon,
            execution_end
        )
        self.assertTrue(
            np.all(
                profile[:, 2] <= 0.,
            ),
            'Indirect price impact should be non-positive when agent has  impact only on opposite events',
        )


if __name__ == '__main__':
    unittest.main()
