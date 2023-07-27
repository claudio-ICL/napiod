import unittest
import numpy as np
import bisect
import scipy
from mpoints import hybrid_hawkes_exp
from napiod import impact


class ImpactTest(unittest.TestCase):
    def setUp(self):
        number_of_event_types: int = 5
        de = number_of_event_types
        number_of_states: int = 9
        dx = number_of_states
        events_labels = [chr(65 + n) for n in range(number_of_event_types)]
        states_labels = [chr(48 + n) for n in range(number_of_states)]
        _phis = [np.eye(dx) + scipy.sparse.random(dx, dx,
                                                  density=.75).A for _ in range(de)]
        _phis = [_phi / np.sum(_phi, axis=1, keepdims=True)
                 for _phi in _phis]
        _phis = [np.expand_dims(_phi, axis=1) for _phi in _phis]
        phis = np.concatenate(_phis, axis=1)
        nus = np.random.uniform(low=0., high=1., size=(de,))
        alphas = .5 * \
            scipy.sparse.random(de, dx*de, density=.50).A.reshape(de, dx, de)
        betas = np.random.uniform(low=.95, high=1.05, size=(de, dx, de))
        hhe = hybrid_hawkes_exp.HybridHawkesExp(
            number_of_event_types,
            number_of_states,
            events_labels,
            states_labels,
        )
        hhe.set_transition_probabilities(phis)
        hhe.set_hawkes_parameters(nus, alphas, betas)
        time_start = 0.
        time_end = 10000.
        times, events, states = hhe.simulate(
            time_start, time_end, max_number_of_events=1000000)
        self.hhe = hhe
        self.number_of_event_types = number_of_event_types
        self.number_of_states = number_of_states
        self.phis = phis
        self.nus = nus
        self.alphas = alphas
        self.betas = betas
        self.times = times
        self.events = events
        self.states = states

    def test_pip(self):
        direction = -1
        j = np.random.randint(low=len(self.times) // 2, high=len(self.times))
        horizon = self.times[j]
        execution_end = self.times[-1]
        dt = self.times[0] / 2.
        self._test_pip(
            direction,
            horizon,
            execution_end,
            dt,
        )

    def _test_pip(self,
                  direction,
                  horizon,
                  execution_end,
                  dt):

        pip = impact.price_impact_profile(
            direction,
            horizon,
            execution_end,
            dt,
            self.phis,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states,
        )

        idx_half = len(pip) // 2
        half_horizon = pip[idx_half, 0]

        half = impact.price_impact_profile(
            direction,
            half_horizon,
            execution_end,
            dt,
            self.phis,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states,
        )
#         print(f'pip.shape: {pip.shape}')
#         print(f'half.shape: {half.shape}')
        self.assertEqual(
            pip.shape,
            (len(pip), 4)
        )
        self.assertEqual(
            half.shape,
            (1 + idx_half, 4)
        )
        self.assertTrue(
            np.all(
                np.diff(pip[:, 0]) > 0.
            ),
            '\nTimes in the price impact profiles are not increasing!'
        )
        self.assertTrue(
            np.all(
                np.diff(pip[:, 1]) >= 0.
            ),
            '\nThe profile of the direct price impact is decreasing!'
        )
        self.assertTrue(
            np.allclose(
                pip[:, 3],
                pip[:, 1] + pip[:, 2],
                rtol=1e-6,
                atol=1e-12,
            ),
            '\nOverall impact is not the sum of direct and indirect impacts'
        )
        self.assertTrue(
            np.allclose(
                half,
                pip[:len(half), :],
                rtol=1e-6,
                atol=1e-12,
            )
        )
        t1 = bisect.bisect_right(pip[:, 0], execution_end)
        self.assertTrue(
            np.allclose(
                np.diff(pip[t1:, 1]),
                0.
            ),
            'Direct impact is not flat after execution'
        )


if __name__ == '__main__':
    unittest.main()
