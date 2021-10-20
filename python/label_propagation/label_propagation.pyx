# cython: language_level=3
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint64_t, uint32_t
from libc.stdio cimport printf

from libcpp cimport bool

from cython.view cimport array as cvarray
import numpy as _np

cimport label_propagation.label_propagation as clp

def compute_ift(features, seeds, neighborhood_size=3, opf_certainty=None):

    features = _np.ascontiguousarray(features).astype(_np.float32)
    seeds = _np.ascontiguousarray(seeds).astype(_np.uint64)

    cdef float *opf_certainty_ptr = NULL
    cdef float [::1] opf_certainty_view

    if opf_certainty is not None:
        opf_certainty = _np.ascontiguousarray(opf_certainty).astype(_np.float32)
        opf_certainty_view = opf_certainty.flatten()

    shape = features.shape
    height = shape[0]
    width = shape[1]
    channels = shape[2]

    cdef float [::1] features_view = features.flatten()
    cdef uint64_t [::1] seeds_view = seeds.flatten()

    pred_out = _np.zeros(height * width, dtype=_np.uint64, order='C')
    root_out = _np.zeros(height * width, dtype=_np.uint64, order='C')
    cost_out = _np.zeros(height * width, dtype=_np.float64, order='C')

    cdef uint64_t [::1] pred_out_view = pred_out
    cdef uint64_t [::1] root_out_view = root_out
    cdef double [::1] cost_out_view = cost_out

    clp.compute_ift(&features_view[0],
                height,
                width,
                &seeds_view[0],
                &opf_certainty_view[0],
                height * width,
                channels,
                neighborhood_size,
                &pred_out_view[0],
                &root_out_view[0],
                &cost_out_view[0])


    cdef uint64_t *labels_ptr = clp.propagate_labels(height,
                                            width,
                                            &seeds_view[0],
                                            &root_out_view[0])

                                            
    cdef double *certainty_ptr = clp.compute_certainty(height,
                                            width,
                                            &cost_out_view[0],
                                            labels_ptr,
                                            &root_out_view[0],
                                            &features_view[0],
                                            channels,
                                            neighborhood_size)

    cdef uint64_t[:, ::1] labels_view = <uint64_t[:height, :width:1]> labels_ptr
    cdef double[:, ::1] certainty_view = <double[:height, :width:1]> certainty_ptr

    labels = _np.asarray(labels_view, dtype=_np.uint64)
    certainty = _np.asarray(certainty_view, dtype=_np.float64)

    return labels, certainty, pred_out, root_out, cost_out


