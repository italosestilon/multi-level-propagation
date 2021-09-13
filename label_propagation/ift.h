# pragma once
#include <cstdint>

#include <vector>

using namespace std;

void compute_itf(const float *features,
                 uint32_t height,
                 uint32_t width,
                 const uint64_t *seeds,
                 float* opf_certainty,
                 uint64_t n_nodes,
                 uint64_t n_features,
                 uint64_t *pred_out,
                 uint64_t *root_out,
                 double *cost_out);

uint64_t *propagate_labels(uint32_t height,
                           uint32_t width,
                           const uint64_t *seeds,
                           const uint64_t *root);

double *compute_certainty(uint32_t height,
                          uint32_t width,
                          double *cost,
                          uint64_t *labels,
                          uint64_t *root,
                          float *features,
                          uint64_t n_features);

vector<uint64_t> neighborhood(uint64_t pixel, uint64_t height, uint64_t width, uint32_t rad=3);