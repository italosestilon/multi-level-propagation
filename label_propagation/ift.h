# pragma once
#include <cstdint>

#include <vector>

using namespace std;

void compute_itf(float *features,
                 uint32_t height,
                 uint32_t width,
                 uint64_t *seeds,
                 uint64_t n_nodes,
                 uint64_t n_features,
                 uint64_t *pred_out,
                 uint64_t *root_out,
                 float *cost_out,
                 bool *visited_out);

float *compute_certainty(uint32_t height, uint32_t width, uint64_t *root, uint64_t *labels);

vector<uint64_t> neighborhood(uint64_t pixel, uint64_t width, uint64_t height);