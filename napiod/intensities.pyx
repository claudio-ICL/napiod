# cython: boundscheck=False, wraparound=False, nonecheck=False

from libc.math cimport log
from libc.math cimport exp
import bisect
import numpy as np
cimport numpy as np
# from libc.stdlib cimport rand, RAND_MAX, srand

DTYPEf = np.float64
DTYPEi = np.int64
DTYPEb = np.uint8
ctypedef np.float64_t DTYPEf_t
ctypedef np.int64_t DTYPEi_t
ctypedef np.uint8_t DTYPEb_t


def impact(
    int direction,
    double t,
    double tau,
    double dt,
    np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
    np.ndarray[DTYPEf_t, ndim=1] nus,
    np.ndarray[DTYPEf_t, ndim=3] alphas,
    np.ndarray[DTYPEf_t, ndim=3] betas,
    np.ndarray[DTYPEf_t, ndim=1] times,
    np.ndarray[DTYPEi_t, ndim=1] events,
    np.ndarray[DTYPEi_t, ndim=1] states,
):
    assert t >= 0.
    assert dt > 0.
    cdef double dt3 = dt / 3.
    cdef double nu = nus[0]
    cdef Py_ssize_t max_num_eval_points = 10 + int(max(t, times[-1]) / dt) + len(times)
    cdef np.ndarray[DTYPEf_t, ndim= 2] imp = np.zeros(
        (max_num_eval_points, 3), dtype=DTYPEf)
    if t < times[0]:
        return imp[:1, :]
    cdef Py_ssize_t de = alphas.shape[0]
    cdef Py_ssize_t dx = alphas.shape[1]
    cdef Py_ssize_t dx3 = dx // 3
    cdef np.ndarray[DTYPEf_t, ndim= 3] acc = np.zeros_like(alphas)
    cdef double direct_phi = .0
    cdef np.ndarray[DTYPEf_t, ndim= 1] indirect_phis = np.zeros(de-1, dtype=DTYPEf)
    cdef Py_ssize_t n = 0
    cdef Py_ssize_t e1, x1, e_
    cdef Py_ssize_t j = 0
    cdef double r = 0.
    cdef double s = 0.
    cdef double s_ = 0.
    cdef long sign
    if direction < 0:
        sign = -1
    else:
        sign = 1
    cdef int is_new_jump_time, is_integration_event
    cdef double t0 = times[0]
    while s + dt3 < t0:
        s += dt
        n += 1
    if t + dt3 < t0:
        return imp[:n, :]
    s = t0
    is_new_jump_time = 1
    while (r < s) & (r < t) & (n < -1 + max_num_eval_points):
        # calcs
        state = states[j]
        is_integration_event = (events[j] == 0)
        for e1 in range(de):
            for x1 in range(dx):
                for e_ in range(de):
                    acc[e1, x1, e_] *= exp(-betas[e1, x1, e_] * (s-r))
                    if is_new_jump_time & is_integration_event:
                        if < long > x1 == state:
                            acc[e1, x1, e_] += 1.

        if is_new_jump_time:
            # If it's new jump time, we have a new state,
            # so we need to update the phis
            if direction < 0:
                direct_phi = np.sum(
                    transition_probabilities[state, 0, :dx3])
            else:
                direct_phi = np.sum(
                    transition_probabilities[state, 0, -dx3:])

            indirect_phis = <double > sign * (
                np.sum(transition_probabilities[state, 1:, -dx3:], axis=1) -
                np.sum(transition_probabilities[state, 1:, :dx3], axis=1)
            )

        imp[n, 0] = s
        if s <= tau:
            imp[n, 1] = direct_phi * (nu + np.sum(
                np.sum(
                    alphas[1:, :, 0] * acc[1:, :, 0],
                    axis=1
                )
            ))
        imp[n, 2] = np.sum(
            indirect_phis * np.sum(
                alphas[0, :, 1:] * acc[0, :, 1:],
                axis=0
            )
        )
        # update indexes
        if s + dt + dt3 < times[j+1]:
            s_ = s + dt
            is_new_jump_time = 0
            if t <= s_:
                s_ = t
        else:
            s_ = times[j+1]
            if t < s_:
                s_ = t
                is_new_jump_time = 0
            else:
                is_new_jump_time = 1
                j += 1

        r = s
        s = s_
        n += 1
    return imp[:n, :]


def _accumulator(
    DTYPEi_t e,
    DTYPEi_t x,
    double t0,
    double t1,
    np.ndarray[DTYPEf_t, ndim=3] betas,
    np.ndarray[DTYPEf_t, ndim=1] times,
    np.ndarray[DTYPEi_t, ndim=1] events,
    np.ndarray[DTYPEi_t, ndim=1] states,
    np.ndarray[DTYPEf_t, ndim=3] acc,
):
    assert t0 <= t1
    cdef Py_ssize_t j0 = bisect.bisect_left(times, t0)
    cdef Py_ssize_t j1 = 1 + bisect.bisect_left(times, t1, lo=j0)
    cdef np.ndarray[DTYPEb_t, ndim= 1] idx = np.logical_and(
        np.logical_and(
            events[j0:j1] == e,
            states[j0:j1] == x,
        ),
        times[j0:j1] <= t1,
    )
    cdef np.ndarray[DTYPEf_t, ndim= 4] eval_times = times[j0:j1][idx].reshape(1, 1, 1, -1)
    cdef np.ndarray[DTYPEf_t, ndim= 4] b = np.expand_dims(betas, axis=3)
    acc *= np.exp(-betas * (t1 - t0))
    acc += np.sum(np.exp(-b * (t1 - eval_times)), axis=3)
    return acc


def direct_impact_spot_intensity(
    int direction,
    double t,
    double tau,
    np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
    np.ndarray[DTYPEf_t, ndim=1] nus,
    np.ndarray[DTYPEf_t, ndim=3] alphas,
    np.ndarray[DTYPEf_t, ndim=3] betas,
    np.ndarray[DTYPEf_t, ndim=1] times,
    np.ndarray[DTYPEi_t, ndim=1] events,
    np.ndarray[DTYPEi_t, ndim=1] states,
):
    if tau < t:
        return 0.
    cdef Py_ssize_t dx = betas.shape[1]
    cdef np.ndarray[DTYPEf_t, ndim= 3] acc = np.zeros_like(alphas)
    cdef double intensity = .0
    for x1 in range(dx):
        acc = np.zeros_like(alphas)
        intensity += np.sum(
            alphas[1:, x1, 0] * _accumulator(
                0,
                x1,
                0.,
                t,
                betas,
                times,
                events,
                states,
                acc
            )[1:, x1, 0]
        )
    intensity += nus[0]
    cdef Py_ssize_t current_idx = max(0, -1 + bisect.bisect_right(times, t))
    cdef int state = states[current_idx]
    cdef Py_ssize_t dx3 = dx // 3
    cdef double phi = .0
    if direction < 0:
        phi = np.sum(transition_probabilities[state, 0, :dx3])
    elif direction > 0:
        phi = np.sum(transition_probabilities[state, 0, -dx3:])
    else:
        raise ValueError(0)
    return phi * intensity


def indirect_impact_spot_intensity(
    int direction,
    double t,
    np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
    np.ndarray[DTYPEf_t, ndim=1] nus,
    np.ndarray[DTYPEf_t, ndim=3] alphas,
    np.ndarray[DTYPEf_t, ndim=3] betas,
    np.ndarray[DTYPEf_t, ndim=1] times,
    np.ndarray[DTYPEi_t, ndim=1] events,
    np.ndarray[DTYPEi_t, ndim=1] states,
):
    cdef Py_ssize_t de = betas.shape[0]
    cdef Py_ssize_t dx = betas.shape[1]
    cdef int sign = 0
    if direction < 0:
        sign = -1
    elif direction > 0:
        sign = 1
    else:
        raise ValueError(0)
    cdef Py_ssize_t current_idx = max(0,
                                      -1 + bisect.bisect_right(times, t)
                                      )
    cdef int state = states[current_idx]
    cdef Py_ssize_t dx3 = dx // 3
    cdef np.ndarray[DTYPEf_t, ndim= 1] phis = np.zeros(de-1, dtype=DTYPEf)
    phis = sign * (
        np.sum(transition_probabilities[state, 1:, -dx3:], axis=1) -
        np.sum(transition_probabilities[state, 1:, :dx3], axis=1)
    )
    cdef np.ndarray[DTYPEf_t, ndim= 3] acc = np.zeros_like(alphas)
    cdef np.ndarray[DTYPEf_t, ndim= 1] intensities = np.zeros(de-1, dtype=DTYPEf)
    for x1 in range(dx):
        acc = np.zeros_like(alphas)
        acc = _accumulator(
            0,
            x1,
            0.,
            t,
            betas,
            times,
            events,
            states,
            acc
        )
        intensities += alphas[0, x1, 1:] * acc[0, x1, 1:]
    cdef double impact = np.sum(phis * intensities)
    return impact


##########################################################################################
# The functions below are used for reconciliation and testing
##########################################################################################
cdef void _acc_step(
    Py_ssize_t j,
    Py_ssize_t de,
    Py_ssize_t dx,
    long e,
    long x,
    double t0,
    double t1,
    double[:, :, :] betas,
    double[:] times,
    long[:] events,
    long[:] states,
    double[:, :, :] acc,
) nogil:
    cdef Py_ssize_t e1, x1, e_
    cdef double tj = times[j]
    cdef int is_integration_event = (events[j] == e)
    cdef int is_integration_state = (states[j] == x)
    cdef int is_jump_time = (t0 < tj) & (tj <= t1)
    cdef int add_to_integral = is_jump_time & is_integration_event & is_integration_state
    for e1 in range(de):
        for x1 in range(dx):
            for e_ in range(de):
                acc[e1, x1, e_] *= exp(-betas[e1, x1, e_] * (t1 - t0))
                if add_to_integral:
                    acc[e1, x1, e_] += exp(-betas[e1, x1, e_] * (t1 - tj))


cdef void _acc_step_for_impact(
    Py_ssize_t j,
    Py_ssize_t de,
    Py_ssize_t dx,
    double r,
    double s,
    double[:, :, :] betas,
    double[:] times,
    long[:] events,
    long[:] states,
    double[:, :, :] acc,
) nogil:
    # This function is meant to mirror the update to the accumulator that happens during the computation of the impact intensities.
    cdef Py_ssize_t e1, x1, e_
    cdef int is_jump_time = (r < times[j]) & (times[j] <= s)
    cdef int is_integration_event = (events[j] == 0)
    cdef long state = states[j]
    for e1 in range(de):
        for x1 in range(dx):
            for e_ in range(de):
                acc[e1, x1, e_] *= exp(-betas[e1, x1, e_] * (s-r))
                if is_jump_time & is_integration_event:
                    if < long > x1 == state:
                        acc[e1, x1, e_] += exp(-betas[e1, x1, e_]
                                               * (s - times[j]))


def _acc_step_for_impact_recon(
        double t0,
        double t1,
        np.ndarray[DTYPEf_t, ndim=3] betas,
        np.ndarray[DTYPEf_t, ndim=1] times,
        np.ndarray[DTYPEi_t, ndim=1] events,
        np.ndarray[DTYPEi_t, ndim=1] states,
        np.ndarray[DTYPEf_t, ndim=3] acc,
):
    assert t0 <= t1
    if not t0 < t1:
        return acc
    cdef Py_ssize_t j0 = bisect.bisect_left(times, t0)
    cdef Py_ssize_t j1 = bisect.bisect_right(times, t1)
    cdef Py_ssize_t n = len(times)
    cdef Py_ssize_t de = betas.shape[0]
    cdef Py_ssize_t dx = betas.shape[1]
    cdef Py_ssize_t j = j0
    cdef double s = t0
    cdef double t = min(t1, times[j0])
    assert s < t
    while (s < t) & (s < t1) & (j <= j1):
        assert t <= t1
        _acc_step_for_impact(
            j,
            de,
            dx,
            s,
            t,
            betas,
            times,
            events,
            states,
            acc
        )
        s = t
        t = min(t1, times[j+1])
        j += 1
    return acc


def _acc_step_recon(
    DTYPEi_t e,
    DTYPEi_t x,
    double t0,
    double t1,
    np.ndarray[DTYPEf_t, ndim=3] betas,
    np.ndarray[DTYPEf_t, ndim=1] times,
    np.ndarray[DTYPEi_t, ndim=1] events,
    np.ndarray[DTYPEi_t, ndim=1] states,
    np.ndarray[DTYPEf_t, ndim=3] acc,
):
    assert t0 <= t1
    if not t0 < t1:
        return acc
    cdef Py_ssize_t j0 = bisect.bisect_left(times, t0)
    cdef Py_ssize_t j1 = bisect.bisect_right(times, t1)
    cdef Py_ssize_t n = len(times)
    cdef Py_ssize_t de = betas.shape[0]
    cdef Py_ssize_t dx = betas.shape[1]
    cdef Py_ssize_t j = j0
    cdef double s = t0
    cdef double t = min(t1, times[j0])
    assert s < t
    while (s < t) & (s < t1) & (j <= j1):
        assert t <= t1
        _acc_step(
            j,
            de,
            dx,
            e,
            x,
            s,
            t,
            betas,
            times,
            events,
            states,
            acc
        )
        s = t
        t = min(t1, times[j+1])
        j += 1
    return acc


def _intensities_for_recon(
    double t,
    np.ndarray[DTYPEf_t, ndim=1] nus,
    np.ndarray[DTYPEf_t, ndim=3] alphas,
    np.ndarray[DTYPEf_t, ndim=3] betas,
    np.ndarray[DTYPEf_t, ndim=1] times,
    np.ndarray[DTYPEi_t, ndim=1] events,
    np.ndarray[DTYPEi_t, ndim=1] states,
):
    # This is an inefficient way to compute intensities.
    # It is only used to test that the function `_accumulator`
    # is consistent with the method
    # `intensities_of_events_at_times` of `HybridHawkesExp`.
    assert t > 0
    cdef Py_ssize_t de = betas.shape[0]
    cdef Py_ssize_t dx = betas.shape[1]
    cdef Py_ssize_t e1 = 0
    cdef Py_ssize_t x1 = 0
    cdef np.ndarray[DTYPEf_t, ndim= 1] lambdas = np.array(nus, copy=True)
    for e1 in range(de):
        for x1 in range(dx):
            acc = np.zeros((de, dx, de), dtype=DTYPEf)
            lambdas += alphas[e1, x1, :] * _accumulator(
                e1, x1, 0, t, betas, times, events, states, acc)[e1, x1, :]
    return lambdas
