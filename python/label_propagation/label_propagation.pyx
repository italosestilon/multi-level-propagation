cimport label_propagation as lp
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t, uint32_t

from libcpp cimport bool

from cython.view cimport array as cvarray
import numpy as np

__all__ = ['py_compute_ift']

def compute_itf(features, seeds, opf_certainty=None):

    assert features.is_c_contiguous(), "features must be c-contiguous"
    assert seeds.is_c_contiguous(), "seeds must be c-contiguous"
    
    shape = features.shape
    height = shape[0]
    width = shape[1]
    channels = shape[2]

    cdef float [:, :, ::1] features_view = features
    cdef uint64_t [:, ::1] seeds_view = seeds
    cdef float [:, ::1] opf_certainty_view = None

    pred_out = np.zeros((height, width), dtype=np.uint64, order='C')
    root_out = np.zeros((height, width), dtype=np.uint64, order='C')
    cost_out = np.zeros((height, width), dtype=np.float64, order='C')
    visited_out = np.zeros((height, width), dtype=np.bool, order='C')

    cdef uint64_t [:, ::1] pred_out_view = pred_out
    cdef uint64_t [:, ::1] root_out_view = root_out
    cdef double [:, ::1] cost_out_view = cost_out
    cdef bool [:, ::1] visited_out_view = visited_out

    lp.compute_itf(&features_view[0, 0, 0],
                height,
                width,
                &seeds_view[0, 0],
                &opf_certainty_view[0, 0],
                height * width,
                channels,
                &pred_out_view[0, 0],
                &root_out_view[0, 0],
                &cost_out_view[0, 0],
                &visited_out_view[0, 0])

    return pred_out, root_out, cost_out



