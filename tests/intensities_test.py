import unittest
import numpy as np
import scipy
from mpoints import hybrid_hawkes_exp
from napiod import intensities


class IntensitiesTest(unittest.TestCase):
    def setUp(self):
        number_of_event_types: int = 5
        de = number_of_event_types
        number_of_states: int = 9
        dx = number_of_states
        events_labels = [chr(65 + n) for n in range(number_of_event_types)]
        states_labels = [chr(48 + n) for n in range(number_of_states)]
        _phis = [np.eye(dx) + scipy.sparse.random(dx, dx,
                                                  density=.50).A for _ in range(de)]
        _phis = [_phi / 10*np.sum(_phi, axis=1, keepdims=True)
                 for _phi in _phis]
        _phis = [np.expand_dims(_phi, axis=1) for _phi in _phis]
        phis = np.concatenate(_phis, axis=1)
        nus = np.random.uniform(low=0., high=1., size=(de,))
        alphas = .5 * \
            scipy.sparse.random(de, dx*de, density=.50).A.reshape(de, dx, de)
        betas = np.ones((de, dx, de), dtype=float)
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
        self.nus = nus
        self.alphas = alphas
        self.betas = betas
        self.times = times
        self.events = events
        self.states = states

    def test_accumulator_calc(self):
        e: int = np.random.randint(low=0, high=self.number_of_event_types)
        x: int = np.random.randint(low=0, high=self.number_of_states)
        t0: float = .0
        t1: float = self.times[-1]
        acc = np.zeros_like(self.alphas)
        acc = intensities._accumulator(
            e,
            x,
            t0,
            t1,
            self.betas,
            self.times,
            self.events,
            self.states,
            acc,
        )
        expected_shape = (
            self.hhe.number_of_event_types,
            self.hhe.number_of_states,
            self.hhe.number_of_event_types
        )
        self.assertEqual(
            acc.shape,
            expected_shape
        )
        napiod_lambdas = intensities._intensities_for_recon(
            t1,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states
        )
        _, mpoints_lambdas = self.hhe.intensities_of_events_at_times(
            np.array([t1]),
            self.times,
            self.events,
            self.states)
        mpoints_lambdas = np.squeeze(mpoints_lambdas)
        print(f'NAPIOD intensities:\n{napiod_lambdas}')
        print(f'MPOINTS intensities:\n{mpoints_lambdas}')
        self.assertTrue(
            np.allclose(
                napiod_lambdas,
                mpoints_lambdas,
                rtol=1e-4,
                atol=1e-8,
            )
        )


if __name__ == '__main__':
    unittest.main()
