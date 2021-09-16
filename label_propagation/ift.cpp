#include <limits>
#include <vector>
#include <label_propagation/ift.h>
#include <label_propagation/priority_queue.h>
#include <label_propagation/utils.h>

using namespace std;

// function to compute neihborhood of a pixel
vector<uint64_t> neighborhood(uint64_t pixel, uint64_t height, uint64_t width, uint32_t rad) {
    // unravel pixel index
    uint64_t x = pixel / width;
    uint64_t y = pixel % width;

    int32_t start = -(rad / 2);
    int32_t end = rad / 2;

    // compute neihborhood
    vector<uint64_t> neighbors;
    for (int32_t i = start; i <= end; i++) {
        for (int32_t j = start; j <= end; j++) {
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

void compute_itf(const float *features,
                uint32_t height,
                uint32_t width,
                const uint64_t *seeds,
                float *opf_certainty,
                uint64_t n_nodes,
                uint64_t n_features,
                uint64_t *pred_out,
                uint64_t *root_out,
                double *cost_out) {

    PairPQ pq = PairPQ(n_nodes);

   // initialize pred, root and cost

   for (uint64_t i = 0; i < n_nodes; i++) {
       if (seeds[i] != 0) {
           pred_out[i] = i;
           root_out[i] = i;
           cost_out[i] = 0;
           if (opf_certainty) {
               cost_out[i] = 1.0 - opf_certainty[i];
           }
       } else {
           pred_out[i] = i;
           root_out[i] = i;

           cost_out[i] = std::numeric_limits<double>::max();

       }
       pq.push(make_pair(i, cost_out[i]));
   }

   while (!pq.empty()) {
       pif first = pq.pop();

       auto neighbors = neighborhood(first.first, height, width, 3);

       double dist;
       double cost;

       for (int i = 0; i < neighbors.size(); i++) {
           auto neighbor = neighbors[i];

           //if (cost_out[neighbor] == 0.0)
           //    continue;

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

double *compute_certainty(uint32_t height,
                          uint32_t width,
                          double *cost,
                          uint64_t *labels,
                          uint64_t *root,
                          float *features,
                          uint64_t n_features) {

    double *certainty = new double[height * width];
    // # pragma omp parallel for
    for (uint64_t i = 0; i < height * width; i++) {
        auto neighbors = neighborhood(i, height, width, 3);
        double min_cost = std::numeric_limits<double>::max();
        // if it is a seed, set certainty to 1
        if (root[i] == i) {
            certainty[i] = 1.0;
        } else {
            for (auto neighbor : neighbors) {
        
                if (labels[neighbor] != labels[i]) {
                    double dist = euclidean_distance(features + i * n_features,
                                                features + neighbor * n_features,
                                                n_features);

                    double alter_cost = max(cost[neighbor], dist);

                    if (min_cost > alter_cost) {
                        min_cost = alter_cost;
                    }
                }
            }

            certainty[i] = min_cost/(cost[i] + min_cost);

        }
    }

    return certainty;
}

uint64_t *propagate_labels(uint32_t height,
                           uint32_t width,
                           const uint64_t *seeds,
                           const uint64_t *root) {
    
    uint64_t *labels = new uint64_t[height * width];

    #pragma omp parallel for
    for (uint64_t i = 0; i < height * width; i++) {
        labels[i] = seeds[root[i]];
    }

    return labels;

}