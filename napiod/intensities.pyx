#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
import bisect
from libc.math cimport exp
from libc.math cimport log
# from libc.stdlib cimport rand, RAND_MAX, srand

DTYPEf = np.float64
DTYPEi = np.int64
ctypedef np.float64_t DTYPEf_t
ctypedef np.int64_t DTYPEi_t

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
    assert t0 < t1
    cdef Py_ssize_t de = betas.shape[0]
    cdef Py_ssize_t dx = betas.shape[1]
    cdef Py_ssize_t i0 = bisect.bisect_right(times, t0)
    cdef Py_ssize_t i1 = 1 +  bisect.bisect_left(times, t1, lo=i0)
    idx = np.logical_and(
            events[i0:i1] == e,
            states[i0:i1] == x
            )
    cdef np.ndarray[DTYPEf_t, ndim=4] eval_times = times[i0:i1][idx].reshape(1, 1, 1, -1)
    cdef np.ndarray[DTYPEf_t, ndim=4] b = np.expand_dims(betas, axis=3)
    acc *= np.exp(-betas * (t1 - t0))
    acc += np.sum(np.exp(-b * (t1 - eval_times)), axis=3)
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
    cdef np.ndarray[DTYPEf_t, ndim=1] lambdas = np.array(nus, copy=True)
    for e1 in range(de):
        for x1 in range(dx):
            acc = np.zeros((de, dx, de), dtype=DTYPEf)
            lambdas += alphas[e1, x1, :] * _accumulator(
                    e1, x1, 0, t, betas, times, events, states, acc)[e1, x1, :]
    return lambdas


def direct_impact(
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
    cdef np.ndarray[DTYPEf_t, ndim=3] acc = np.zeros_like(alphas)
    cdef double intensity = .0
    for x1 in range(dx):
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
    cdef Py_ssize_t current_idx = bisect.bisect_left(times, t)
    cdef int state = states[current_idx]
    cdef Py_ssize_t dx3 = dx // 3
    cdef double phi = .0
    if direction < 0 :
        phi = np.sum(transition_probabilities[state, 0, :dx3])
    elif direction > 0:
        phi = np.sum(transition_probabilities[state, 0, -dx3:])
    else:
       raise ValueError(0)
    return phi * intensity


def indirect_impact(
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
    cdef Py_ssize_t current_idx = bisect.bisect_left(times, t)
    cdef int state = states[current_idx]
    cdef Py_ssize_t dx3 = dx // 3
    cdef np.ndarray[DTYPEf_t, ndim=1] phis = np.zeros(de-1, dtype=DTYPEf)
    phis = sign * (
            np.sum(transition_probabilities[state, 1:, -dx3:], axis=1) - 
            np.sum(transition_probabilities[state, 1:, :dx3], axis=1) 
            )
    cdef np.ndarray[DTYPEf_t, ndim=3] acc = np.zeros_like(alphas)
    cdef np.ndarray[DTYPEf_t, ndim=1] intensities = np.zeros(de-1, dtype=DTYPEf)
    for x1 in range(dx):
        intensities += alphas[0, x1, 1:] * _accumulator(
                    0, 
                    x1,
                    0., 
                    t,
                    betas,
                    times,
                    events,
                    states,
                    acc
                    )[0, x1, 1:]
    cdef double impact = np.sum(phis * intensities)
    return impact









