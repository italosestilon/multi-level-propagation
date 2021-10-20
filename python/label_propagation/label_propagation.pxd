from libc.stdint cimport uint64_t, uint32_t

cdef extern from "<label_propagation/ift.h>":
    void compute_ift(const float *features,
                uint32_t height,
                uint32_t width,
                const uint64_t *seeds,
                float* opf_certainty,
                uint64_t n_nodes,
                uint64_t n_features,
                uint32_t neighborhood_size,
                uint64_t *pred_out,
                uint64_t *root_out,
                double *cost_out)

    uint64_t *propagate_labels(uint64_t height,
                            uint64_t width,
                            const uint64_t *seeds,
                            const uint64_t *root)

    double* compute_certainty(uint32_t height,
                        uint32_t width,
                        double *cost,
                        uint64_t *labels,
                        uint64_t *root,
                        float *features,
                        uint64_t n_features,
                        uint32_t neighborhood_size)