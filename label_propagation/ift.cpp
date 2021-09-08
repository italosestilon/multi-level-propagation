#include <cmath>
#include <limits>
#include <vector>
#include <label_propagation/ift.h>
#include <label_propagation/priority_queue.h>

using namespace std;

inline float euclidean_distance(const float *v1, const float *v2, unsigned long int dims) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (unsigned long int i = 0; i < dims; i++) {
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return sqrt(sum);
}

// function to compute neihborhood of a pixel
vector<uint64_t> neighborhood(uint64_t pixel, uint64_t height, uint64_t width) {
    // unravel pixel index
    uint64_t x = pixel / width;
    uint64_t y = pixel % width;

    // compute neihborhood
    vector<uint64_t> neighbors;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            // check if neihbor is in image
            if (x + i >= 0 && x + i < width && y + j >= 0 && y + j < height) {
                // add neihbor to neihborhood
                uint64_t neighbor = (x + i) * width + (y + j);
                neighbors.push_back(neighbor);
            }
        }
    }

    return neighbors;

}

void compute_itf(float *features,
                uint32_t height,
                uint32_t width,
                uint64_t *seeds,
                uint64_t n_nodes,
                uint64_t n_features,
                uint64_t *pred_out,
                uint64_t *root_out,
                float *cost_out,
                bool *visited_out) {

    PairPQ pq = PairPQ(n_nodes);

   // initialize pred, root and cost

   for (uint64_t i = 0; i < n_nodes; i++) {
       if (seeds[i] != 0) {
           pred_out[i] = i;
           root_out[i] = i;
           cost_out[i] = 0;
       } else {
           pred_out[i] = numeric_limits<uint64_t>::max();
           root_out[i] = numeric_limits<uint64_t>::max();

           cost_out[i] = std::numeric_limits<float>::max();

       }
       pq.push(make_pair(i, cost_out[i]));
   }

   while (!pq.empty()) {
       pif first = pq.pop();

       visited_out[first.first] = true;

       auto neighbors = neighborhood(first.first, height, width);

       float dist;
       float cost;

       for (int i = 0; i < neighbors.size(); i++) {
           auto neighbor = neighbors[i];

           if (cost_out[neighbor] == 0.0)
               continue;

           dist = euclidean_distance(features + first.first * n_features,
                                     features + neighbor * n_features,
                                     n_features);

           cost = max(first.second, dist);
           if (cost < cost_out[neighbor]) {
               cost_out[neighbor] = cost;
               pred_out[neighbor] = first.first;
               root_out[neighbor] = root_out[first.first];
               pq.decrease_key(pq.get_index(neighbor), cost);

           }
       }
   }
}

float *compute_certainty(uint32_t height, uint32_t width, uint64_t *root, uint64_t *labels) {
    float *certainty = new float[height * width];
    // # pragma omp parallel for
    for (uint64_t i = 0; i < height * width; i++) {
        auto neighbors = neighborhood(i, height, width);
        float certainty_sum = 0;
        if (root[i] == i) {
            certainty_sum = 1.0;
        } else {
            for (auto neighbor : neighbors) {
                if (labels[neighbor] == labels[i]) {
                    certainty_sum += 1/((float)neighbors.size());
                }
            }
        }

        certainty[i] = certainty_sum;
    }

    return certainty;
}