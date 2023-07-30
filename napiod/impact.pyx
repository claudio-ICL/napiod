# cython: boundscheck=False, wraparound=False, nonecheck=False

from napiod import intensities
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


def price_impact_profile(
    int direction,
    double horizon,
    double execution_end,
    double dt,
    np.ndarray[DTYPEf_t, ndim=3] transition_probabilities,
    np.ndarray[DTYPEf_t, ndim=1] nus,
    np.ndarray[DTYPEf_t, ndim=3] alphas,
    np.ndarray[DTYPEf_t, ndim=3] betas,
    np.ndarray[DTYPEf_t, ndim=1] times,
    np.ndarray[DTYPEi_t, ndim=1] events,
    np.ndarray[DTYPEi_t, ndim=1] states,
):
    cdef np.ndarray[DTYPEf_t, ndim = 2] f = intensities.impact(
        direction,
        horizon,
        execution_end,
        dt,
        transition_probabilities,
        nus,
        alphas,
        betas,
        times,
        events,
        states
    )
    assert len(f) > 1
    assert f.shape[1] == 3
    cdef Py_ssize_t n = len(f) - 1
    cdef np.ndarray[DTYPEf_t, ndim = 2] pip = np.zeros((n, 4), dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim = 1] delta_t = np.diff(f[:, 0])
    pip[:, 0] = f[1:, 0]
    pip[:, 1] = np.cumsum(.5 * (f[1:, 1] + f[:n, 1]) * delta_t)
    pip[:, 2] = np.cumsum(.5 * (f[1:, 2] + f[:n, 2]) * delta_t)
    pip[:, 3] = pip[:, 1] + pip[:, 2]
    return pip
