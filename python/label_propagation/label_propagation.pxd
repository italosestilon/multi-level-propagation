from libc.stdint cimport uint64_t, uint32_t
from libcpp cimport bool

cdef extern from "label_propagation/label_propagation.h":
    void compute_itf(const float *features,
                 uint32_t height,
                 uint32_t width,
                 const uint64_t *seeds,
                 float* opf_certainty,
                 uint64_t n_nodes,
                 uint64_t n_features,
                 uint64_t *pred_out,
                 uint64_t *root_out,
                 double *cost_out,
                 bool *visited_out)