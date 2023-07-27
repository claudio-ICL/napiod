import unittest
import bisect
import numpy as np
import scipy
from mpoints import hybrid_hawkes_exp
from napiod import intensities


class IntensitiesTest(unittest.TestCase):
    def setUp(self):
        number_of_event_types: int = 4
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
        # Test that accumulator returns an array of the correct shape
        expected_shape = (
            self.hhe.number_of_event_types,
            self.hhe.number_of_states,
            self.hhe.number_of_event_types
        )
        self.assertEqual(
            acc.shape,
            expected_shape
        )

        # Test that the intensities computed using the accumulator are decreasing between arrival times
        j = len(self.times) // 2
        t_j = self.times[j]
        t_j_next = self.times[j+1]
        tau = self.times[-1]
        s = (t_j + t_j_next) / 2.
        t = (s + t_j_next) / 2.
        lambdas_s = intensities._intensities_for_recon(
            s,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states
        )
        lambdas_t = intensities._intensities_for_recon(
            t,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states
        )
        self.assertTrue(
            np.all(lambdas_t < lambdas_s),
            'Intensities are not decreasing between arrival times'
        )

        # Test consistency with mpoints
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
#         # print(f'NAPIOD intensities:\n{napiod_lambdas}')
#         # print(f'MPOINTS intensities:\n{mpoints_lambdas}')
        self.assertTrue(
            np.allclose(
                napiod_lambdas,
                mpoints_lambdas,
                rtol=1e-4,
                atol=1e-8,
            )
        )

    def test_accumulator_step_at_start(self):
        e: int = self.events[0]
        x: int = self.states[0]
        t0: float = self.times[0] / 2.
        t1: float = (self.times[0] + self.times[1]) / 2.
        self._test_accumulator_step(e, x, t0, t1)
        e = self.number_of_event_types - 1 if e == 0 else e - 1
        x = self.number_of_states - 1 if x == 0 else x - 1
        self._test_accumulator_step(e, x, t0, t1)

    def test_accumulator_step_at_second_start(self):
        e: int = self.events[0]
        x: int = self.states[0]
        t0: float = (self.times[0] + self.times[1]) / 2.
        t1: float = (self.times[1] + self.times[2]) / 2.
        self._test_accumulator_step(e, x, t0, t1)
        e = self.number_of_event_types - 1 if e == 0 else e - 1
        x = self.number_of_states - 1 if x == 0 else x - 1
        self._test_accumulator_step(e, x, t0, t1)

    def test_accumulator_step_random(self):
        e: int = np.random.randint(low=0, high=self.number_of_event_types)
        x: int = np.random.randint(low=0, high=self.number_of_states)
        t1: float = 1e-9 + np.random.choice(self.times[len(self.times) // 2:])
        t0: float = t1 / 2.
        self._test_accumulator_step(e, x, t0, t1)

    def _test_accumulator_step(self, e, x,  t0, t1):

        def recon(s, t, acc_vector, acc_step):
            acc_vector = intensities._accumulator(
                e,
                x,
                s,
                t,
                self.betas,
                self.times,
                self.events,
                self.states,
                acc_vector
            )
            acc_step = intensities._acc_step_recon(
                e,
                x,
                s,
                t,
                self.betas,
                self.times,
                self.events,
                self.states,
                acc_step
            )
#             print(f'acc_vector:\n{acc_vector}')
#             print(f'acc_step:\n{acc_step}')
            error_msg = f'\nacc_vector and acc_step do not reconcile in the increment from s={s} to  t={t}\n'
            error_msg += f'acc_vector:\n{acc_vector}\n\n'
            error_msg += f'acc_step:\n{acc_step}\n\n'
            self.assertTrue(
                np.allclose(
                    acc_vector,
                    acc_step,
                    atol=1e-12,
                    rtol=1e-7,
                ),
                error_msg,
            )
            return acc_vector, acc_step
#         print('\n...Testing accumulator step.....')
#         print(f'e: {e}')
#         print(f'x: {x}')
#         print(f't0: {t0}')
#         print(f't1: {t1}')
        acc_vector = np.zeros_like(self.alphas)
        acc_step = np.zeros_like(self.alphas)
        acc_vector, acc_step = recon(
            0., t0, acc_vector, acc_step)
        acc_vector, acc_step = recon(
            t0, t1, acc_vector, acc_step)

    def test_acc_step_for_impact_at_start(self):
        j = 0
        tj = self.times[j]
        tj_next = self.times[j+1]
        t = (tj + tj_next) / 2.
        self._recon_acc_for_impact_and_acc_for_intensity(t)

    def test_acc_step_for_impact(self):
        j = np.random.randint(len(self.times) - 1)
        tj = self.times[j]
        tj_next = self.times[j+1]
        t = (tj + tj_next) / 2.
        self._recon_acc_for_impact_and_acc_for_intensity(t)

    def _recon_acc_for_impact_and_acc_for_intensity(self, t):
        #         print('.....Recon accumulator for intensity and accumulator for impact....')
        #         print(f't: {t}')
        acc_full = np.zeros_like(self.alphas)
        for x1 in range(self.number_of_states):
            acc = np.zeros_like(self.alphas)
            intensities._acc_step_recon(
                0,
                x1,
                .0,
                t,
                self.betas,
                self.times,
                self.events,
                self.states,
                acc
            )
            acc_full[:, x1, :] = acc[:, x1, :]
        acc_imp = np.zeros_like(self.alphas)
        acc_imp = intensities._acc_step_for_impact_recon(
            .0,
            t,
            self.betas,
            self.times,
            self.events,
            self.states,
            acc_imp
        )
#         print(f'acc_full:\n{acc_full}')
#         print(f'acc_imp:\n{acc_imp}')
        rtol = 1e-7
        atol = 1e-12
        is_close = np.isclose(
            acc_full, acc_imp, rtol=rtol, atol=atol)
        error_msg = 'The accumulator for the impact intensity does not reconcile with the full accumulator\n'
        error_msg += f'acc_full:\n{acc_full}\n\n'
        error_msg += f'acc_imp:\n{acc_imp}\n'
        error_msg += '\nNon-reconciled entries:\n'
        error_msg += f'acc_full:\n{acc_full[~is_close]}\n\n'
        error_msg += f'acc_imp:\n{acc_imp[~is_close]}\n'

        self.assertTrue(
            np.allclose(
                acc_full,
                acc_imp,
                rtol=rtol,
                atol=atol,
            ),
            error_msg
        )

    def _direct_impact_sensitivity_test(
            self,
            direction,
    ):
        j = len(self.times) // 2
        t_j = self.times[j]
        t_j_next = self.times[j+1]
        tau = self.times[-1]
        s = (t_j + t_j_next) / 2.
        t = (s + t_j_next) / 2.
        self.assertEqual(
            intensities.direct_impact_spot_intensity(
                direction,
                t_j_next,
                t_j,
                self.phis,
                self.nus,
                self.alphas,
                self.betas,
                self.times,
                self.events,
                self.states
            ),
            0.,
            'Direct impact after execution is not zero'
        )
        impact_s = intensities.direct_impact_spot_intensity(
            direction,
            s,
            tau,
            self.phis,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states
        )
        impact_t = intensities.direct_impact_spot_intensity(
            direction,
            t,
            tau,
            self.phis,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states
        )
        # print(f'impact_s: {impact_s}')
        # print(f'impact_t: {impact_t}')
        self.assertLessEqual(
            impact_t,
            impact_s,
            'Direct impact is increasing between arrival times'
        )
        low_betas = np.array(self.betas, copy=True)
        low_betas[1:, :, 0] /= 2.
        impact_t_low_betas = intensities.direct_impact_spot_intensity(
            direction,
            t,
            tau,
            self.phis,
            self.nus,
            self.alphas,
            low_betas,
            self.times,
            self.events,
            self.states
        )
        # print(f'impact_t: {impact_t}')
        # print(f'impact_t_low_betas: {impact_t_low_betas}')
        error_msg = 'Decreasing the decay rate of the agent does not decrease the direct impact\n'
        error_msg += f'impact_t: {impact_t}\n'
        error_msg += f'impact_t_low_betas: {impact_t}\n'
        assertion = self.assertLess if impact_t > 0. else self.assertLessEqual
        assertion(
            impact_t,
            impact_t_low_betas,
            error_msg
        )
        phis_ = np.array(self.phis, copy=True)
        if direction < 0:
            phis_[:, 0, :self.number_of_states // 3] = 0.
        else:
            phis_[:, 0, -(self.number_of_states//3):] = 0.
        mass = np.sum(phis_[:, 0, :], axis=1, keepdims=True)
        if not np.all(mass > 0.):
            print(
                'Warning: some of the masses of transitions probabilities associated with event 0 are null')
            return
        phis_[:, 0, :] /= mass
        impact_t_zero_phis = intensities.direct_impact_spot_intensity(
            direction,
            t,
            tau,
            phis_,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states
        )
        # print(f'impact_t_zero_phis: {impact_t_zero_phis}')
        self.assertLess(
            np.abs(impact_t_zero_phis),
            1e-9,
        )

    def test_seller_direct_impact_sensitivity(self):
        self._direct_impact_sensitivity_test(direction=-1)

    def test_buyer_direct_impact_sensitivity(self):
        self._direct_impact_sensitivity_test(direction=1)

    def _indirect_impact_sensitivity_test(
            self,
            direction,
    ):
        j = len(self.times) // 2
        t_j = self.times[j]
        t_j_next = self.times[j+1]
        tau = self.times[-1]
        s = (t_j + t_j_next) / 2.
        t = (s + t_j_next) / 2.
        u = (t_j + self.times[int(1.5 * j)]) / 2.
        impact_s = intensities.indirect_impact_spot_intensity(
            direction,
            s,
            self.phis,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states
        )
        impact_t = intensities.indirect_impact_spot_intensity(
            direction,
            t,
            self.phis,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states
        )
#         print(f'impact_s: {impact_s}')
#         print(f'impact_t: {impact_t}')
        self.assertLessEqual(
            abs(impact_t),
            abs(impact_s),
            'Absolute Indirect impact is increasing between arrival times'
        )
        eps = 1e-8
        phis = eps + np.array(self.phis, copy=True)
        symphi = np.array(phis, copy=True)
        dx3 = self.number_of_states // 3
        inflationary_mass = np.sum(
            phis[:, 1:, -dx3:],
            axis=2,
            keepdims=True)
        deflationary_mass = np.sum(
            phis[:, 1:, :dx3],
            axis=2,
            keepdims=True)
        dm = np.array(deflationary_mass, copy=True)
        im = np.array(inflationary_mass, copy=True)
        symphi[:, 1:, -dx3:] = (
            0.5 * (inflationary_mass + deflationary_mass) *
            symphi[:, 1:, -dx3:] / im
        )
        symphi[:, 1:, :dx3] = (
            0.5 * (inflationary_mass + deflationary_mass) *
            symphi[:, 1:, :dx3] / dm
        )

        sym_inflationary_mass = np.sum(
            symphi[:, 1:, -dx3:],
            axis=2,
            keepdims=True)
        sym_deflationary_mass = np.sum(
            symphi[:, 1:, :dx3],
            axis=2,
            keepdims=True)
#         print(f'Symmetric Deflationary masses: \n {sym_deflationary_mass}')
#         print(f'Symmetric Inflationary masses: \n {sym_inflationary_mass}')
        self.assertTrue(
            np.allclose(
                sym_inflationary_mass,
                sym_deflationary_mass,
                atol=1e-5,
                rtol=1e-3,
            )
        )
        sym_impact_t = intensities.indirect_impact_spot_intensity(
            direction,
            t,
            symphi,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states
        )
        sym_impact_u = intensities.indirect_impact_spot_intensity(
            direction,
            u,
            symphi,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states
        )
#         print(f'sym_impact_t: {sym_impact_t}')
#         print(f'sym_impact_u: {sym_impact_u}')
        self.assertTrue(
            np.allclose(
                sym_impact_u,
                .0,
                atol=1e-12,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            np.allclose(
                sym_impact_t,
                .0,
                atol=1e-12,
                rtol=1e-6,
            )
        )

    def test_buyer_indirect_impact_sensitivity(
            self):
        self._indirect_impact_sensitivity_test(
            1
        )

    def test_seller_indirect_impact_sensitivity(
            self):
        self._indirect_impact_sensitivity_test(
            -1
        )

    def test_seller_impact_trajectory_and_spot_impact(
            self):
        direction = -1
        half = len(self.times) // 2
        j = half + np.random.choice(half)
        tj = self.times[j]
        t = (self.times[j-1] + tj) / 2.
        tau = self.times[-1] + 1e-4
        dt = self.times[0] / 4.
        self._recon_impact_trajectory_and_spot_impact(
            direction,
            t,
            tau,
            dt
        )

    def test_buyer_impact_trajectory_and_spot_impact(
            self):
        direction = 1
        half = len(self.times) // 2
        j = half + np.random.choice(half)
        tj = self.times[j]
        t = (self.times[j-1] + tj) / 2.
        tau = self.times[-1] + 1e-4
        dt = self.times[0] / 4.
        self._recon_impact_trajectory_and_spot_impact(
            direction,
            t,
            tau,
            dt
        )

    def test_fine_seller_impact_trajectory_and_spot_impact_at_start_with_no_agent_event(
            self):
        direction = -1
        j = 1
        tj = self.times[j]
        t = (self.times[j-1] + tj) / 2.
        tau = self.times[-1] + 1e-4
        dt = .85 * self.times[0] / 2.
        events = np.array(self.events, copy=True)
        if events[0] == 0:
            events[0] = np.random.choice(
                list(range(1, self.number_of_event_types)))
        self._recon_impact_trajectory_and_spot_impact(
            direction,
            t,
            tau,
            dt,
            events,
        )

    def test_fine_seller_impact_trajectory_and_spot_impact_at_start_with_agent_event(
            self):
        direction = -1
        j = 1
        tj = self.times[j]
        t = (self.times[j-1] + tj) / 2.
        tau = self.times[-1] + 1e-4
        dt = .85 * self.times[0] / 2.
        events = np.array(self.events, copy=True)
        events[0] = 0
        self._recon_impact_trajectory_and_spot_impact(
            direction,
            t,
            tau,
            dt,
            events,
        )

    def test_fine_buyer_impact_trajectory_and_spot_impact_at_start_with_agent_event(
            self):
        direction = 1
        j = 1
        tj = self.times[j]
        t = (self.times[j-1] + tj) / 2.
        tau = self.times[-1] + 1e-4
        dt = .85 * self.times[0] / 2.
        events = np.array(self.events, copy=True)
        events[0] = 0
        self._recon_impact_trajectory_and_spot_impact(
            direction,
            t,
            tau,
            dt,
            events,
        )

    def test_impact_trajectory_and_spot_impact_at_first_10(
            self):
        direction = int(np.random.choice([-1, 1]))
        j = max(1, -1 + min(11, len(self.times)))
        tj = self.times[j]
        t = (self.times[j-1] + tj) / 2.
        tau = self.times[-1] + 1e-4
        dt = self.times[j + 1]
        events = self.events
        self._recon_impact_trajectory_and_spot_impact(
            direction,
            t,
            tau,
            dt,
            events,
        )

    def test_fine_impact_trajectory_and_spot_impact_at_first_10(
            self):
        direction = int(np.random.choice([-1, 1]))
        j = max(1, -1 + min(11, len(self.times)))
        tj = self.times[j]
        t = (self.times[j-1] + tj) / 2.
        tau = self.times[-1] + 1e-4
        dt = float(
            np.random.uniform(low=.5, high=.95) * self.times[0] / 2.
        )
        events = self.events
        self._recon_impact_trajectory_and_spot_impact(
            direction,
            t,
            tau,
            dt,
            events,
        )

    def test_seller_impact_trajectory_and_spot_impact_at_start_with_no_agent_event(
            self):
        direction = -1
        j = 1
        tj = self.times[j]
        t = (self.times[j-1] + tj) / 2.
        tau = self.times[-1] + 1e-4
        dt = self.times[0] + 1.
        events = np.array(self.events, copy=True)
        if events[0] == 0:
            events[0] = np.random.choice(
                list(range(1, self.number_of_event_types)))
        self._recon_impact_trajectory_and_spot_impact(
            direction,
            t,
            tau,
            dt,
            events,
        )

    def test_seller_impact_trajectory_and_spot_impact_at_start_with_agent_event(
            self):
        direction = -1
        j = 1
        tj = self.times[j]
        t = (self.times[j-1] + tj) / 2.
        tau = self.times[-1] + 1e-4
        dt = self.times[0] + 1.
        events = np.array(self.events, copy=True)
        events[0] = 0
        self._recon_impact_trajectory_and_spot_impact(
            direction,
            t,
            tau,
            dt,
            events,
        )

    def test_seller_impact_trajectory_and_spot_impact_at_second_start_with_agent_event(
            self):
        direction = -1
        j = 2
        tj = self.times[j]
        t = (self.times[j-1] + tj) / 2.
        tau = self.times[-1] + 1e-4
        dt = self.times[j] + 1.
        events = np.array(self.events, copy=True)
        events[0] = 0
        self._recon_impact_trajectory_and_spot_impact(
            direction,
            t,
            tau,
            dt,
            events,
        )

    def test_seller_impact_trajectory_and_spot_impact_at_second_start(
            self):
        direction = -1
        j = 2
        tj = self.times[j]
        t = (self.times[j-1] + tj) / 2.
        tau = self.times[-1] + 1e-4
        dt = self.times[j] + 1.
        events = self.events
        self._recon_impact_trajectory_and_spot_impact(
            direction,
            t,
            tau,
            dt,
            events,
        )

    def _recon_impact_trajectory_and_spot_impact(
            self,
            direction,
            t,
            tau,
            dt,
            events=None
    ):
        if events is None:
            events = self.events
        di = intensities.direct_impact_spot_intensity(
            direction,
            t,
            tau,
            self.phis,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            events,
            self.states,
        )
        ii = intensities.indirect_impact_spot_intensity(
            direction,
            t,
            self.phis,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            events,
            self.states,
        )
        impact = intensities.impact(
            direction,
            t,
            tau,
            dt,
            self.phis,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            events,
            self.states,
        )
        tell_time = f't: {t}\n'
        tell_direct_impact = f'Direct impact: {di}\n'
        tell_indirect_impact = f'Indirect impact: {ii}\n'
        tell_last_point = f'len(impact): {len(impact)}; impact[-5:, :]:\n {impact[-5:, :]}\n'
#         print(tell_time)
#         print(tell_direct_impact)
#         print(tell_indirect_impact)
#         print(tell_last_point)
        error_msg = '\nImpact from trajectory does not reconcile with spot impact\n'
        error_msg += tell_time
        error_msg += tell_direct_impact + tell_indirect_impact
        error_msg += tell_last_point
        self.assertTrue(
            np.allclose(
                np.array(
                    [t, di, ii]
                ),
                impact[-1, :],
                rtol=1e-4,
                atol=1e-8,
            ),
            error_msg
        )

    def test_after_sell_execution_decrease(self):
        direction = -1
        t = self.times[-1]
        tau = self.times[len(self.times) // 2]
        self._test_after_execution_decrease(
            direction,
            t,
            tau
        )

    def test_after_buy_execution_decrease(self):
        direction = 1
        t = self.times[-1]
        tau = self.times[len(self.times) // 2]
        self._test_after_execution_decrease(
            direction,
            t,
            tau
        )

    def _test_after_execution_decrease(
            self,
            direction,
            t,
            tau
    ):
        dt = .25 * self.times[0]
        impact = intensities.impact(
            direction,
            t,
            tau,
            dt,
            self.phis,
            self.nus,
            self.alphas,
            self.betas,
            self.times,
            self.events,
            self.states,
        )
        t1 = bisect.bisect_right(impact[:, 0], tau)
        self.assertTrue(
            np.allclose(
                np.diff(impact[t1:, 1]),
                0.
            ),
            'Intensity of direct impact is not null after execution',
        )


if __name__ == '__main__':
    unittest.main()
