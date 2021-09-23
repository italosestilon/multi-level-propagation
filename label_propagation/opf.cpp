#include <label_propagation/opf.h>
#include <label_propagation/priority_queue.h>
#include <label_propagation/utils.h>

#include <queue>
#include <cmath>

Graph *semi_supervised_train(float *features,
        unsigned int *label,
        bool *is_supervised,
        unsigned long int n_samples,
        unsigned long int n_features) {

    Graph *g = new Graph(n_samples, n_features, features);

    PairPQ pq = PairPQ(n_samples);

    // initialize the graph
    for (unsigned long int i = 0; i < n_samples; i++) {
        if (is_supervised[i]) {
            g->get_labels()[i] = label[i];
            g->get_path_value()[i] = 0;
            g->is_supervised()[i] = true;
        } else {
            g->get_path_value()[i] = numeric_limits<float>::infinity();
        }

        pq.push(make_pair(i, g->get_path_value()[i]));
    }
    unsigned long int i = 0;
    while (!pq.empty()) {
        pif p = pq.pop();
        
        unsigned long int node_id = p.first;
        double node_value = p.second;

        g->get_ordered_nodes()[i++] = node_id;
        g->get_weight()[node_id] = node_value;
        double dist = 0;

        for (unsigned long int j = 0; j < n_samples; j++) {
            if (node_value < g->get_path_value()[j]) {
                dist = euclidean_distance(features + node_id, features + j, n_features);
            }

            double candidate_value = max(node_value, dist);

            if (candidate_value < node_value) {
                g->get_path_value()[j] = candidate_value;
                g->get_nodes()[j].set_pred(j);
                g->get_labels()[j] = g->get_labels()[node_id];
            }
        }
    }
    
    return g;
}

pair<uint32_t, double> find_min_dists(Graph *g,
                                    float *features,
                                    uint64_t sample,
                                    uint64_t n_features) {

    double min_dist = numeric_limits<double>::infinity();
    uint32_t label = 0;

    for (uint64_t i = 0; i < g->get_n_nodes(); i++) {
        uint64_t node_id = g->get_ordered_nodes()[i];
        float *node_features = g->get_node_features(node_id);
        double dist = euclidean_distance(node_features,
                                         features + sample,
                                         n_features);

        if (dist < min_dist) {
            min_dist = dist;
            label = g->get_labels()[node_id];
        } 
    }

    return make_pair(label, min_dist);
}

double find_min_dists_with_label(Graph *g,
                                float *features,
                                uint64_t sample,
                                uint64_t n_features,
                                uint32_t label) {
    double min_dist = numeric_limits<double>::infinity();

    for (uint64_t i = 0; i < g->get_n_nodes(); i++) {
        uint64_t node_id = g->get_ordered_nodes()[i];
        uint32_t node_label = g->get_labels()[node_id];

        if (node_label != label) {
            float *node_features = g->get_node_features(node_id);
            double dist = euclidean_distance(node_features, features + sample, n_features);

            if (dist < min_dist) {
                min_dist = dist;
            }
        }
    }

    return min_dist;
}

void classify_with_certainty(Graph *graph,
                             float *features,
                             uint64_t n_samples,
                             uint64_t n_features,
                             uint32_t *labels_out,
                             double *certainties_out) {

    for (uint64_t i = 0; i < n_samples; i++) {
        auto label_and_dist = find_min_dists(graph,
                                             features,
                                             i,
                                             n_features);
        auto label = label_and_dist.first;
        auto dist = label_and_dist.second;

        double dist_to_other_label = find_min_dists_with_label(graph,
                                                               features,
                                                               i,
                                                               n_features,
                                                               label);

        labels_out[i] = label;
        certainties_out[i] = dist_to_other_label/(dist + dist_to_other_label);
    }
}