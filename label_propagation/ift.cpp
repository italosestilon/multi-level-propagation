#include <limits>
#include <vector>
#include <label_propagation/ift.h>
#include <label_propagation/priority_queue.h>
#include <label_propagation/utils.h>

using namespace std;
void compute_ift(const float *features,
                 uint32_t height,
                 uint32_t width,
                 const uint64_t *seeds,
                 float *opf_certainty,
                 uint64_t n_nodes,
                 uint64_t n_features,
                 uint32_t neighborhood_size,
                 uint64_t *pred_out,
                 uint64_t *root_out,
                 double *cost_out)
{

    PairPQ pq = PairPQ(n_nodes);

    // initialize pred, root and cost

    for (uint64_t i = 0; i < n_nodes; i++)
    {
        if (seeds[i] != 0)
        {
            pred_out[i] = i;
            root_out[i] = i;
            cost_out[i] = 0;
            if (opf_certainty)
            {
                cost_out[i] = 1.0 - opf_certainty[i];
            }
        }
        else
        {
            pred_out[i] = i;
            root_out[i] = i;

            cost_out[i] = std::numeric_limits<double>::max();
        }
        pq.push(make_pair(i, cost_out[i]));
    }

    while (!pq.empty())
    {
        pif first = pq.pop();

        auto neighbors = neighborhood(first.first, height, width, neighborhood_size);

        double dist;
        double cost;

        for (uint32_t i = 0; i < neighbors.size(); i++)
        {
            auto neighbor = neighbors[i];

            //if (cost_out[neighbor] == 0.0)
            //    continue;

            dist = euclidean_distance(features + first.first * n_features,
                                      features + neighbor * n_features,
                                      n_features);

            cost = max(first.second, dist);
            if (cost < cost_out[neighbor])
            {
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
                          uint64_t n_features,
                          uint32_t neighborhood_size)
{

    double *certainty = new double[height * width];
    // # pragma omp parallel for
    for (uint64_t i = 0; i < height * width; i++)
    {
        auto neighbors = neighborhood(i, height, width, neighborhood_size);
        double min_cost = std::numeric_limits<double>::max();
        // if it is a seed, set certainty to 1
        if (root[i] == i && cost[i] == 0)
        {
            certainty[i] = 1.0;
        }
        else
        {
            for (auto neighbor : neighbors)
            {

                if (labels[neighbor] != labels[i])
                {
                    double dist = euclidean_distance(features + i * n_features,
                                                     features + neighbor * n_features,
                                                     n_features);

                    double alter_cost = max(cost[neighbor], dist);

                    if (min_cost > alter_cost)
                    {
                        min_cost = alter_cost;
                    }
                }
            }

            certainty[i] = min_cost / (cost[i] + min_cost);
        }
    }

    return certainty;
}

uint64_t *propagate_labels(uint32_t height,
                           uint32_t width,
                           const uint64_t *seeds,
                           const uint64_t *root)
{

    uint64_t *labels = new uint64_t[height * width];

#pragma omp parallel for
    for (uint64_t i = 0; i < height * width; i++)
    {
        labels[i] = seeds[root[i]];
    }

    return labels;
}