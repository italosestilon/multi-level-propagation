#include <label_propagation/opf.h>
#include <queue>
#include <cmath>

typedef pair<int, float> pf;

inline float euclidean_distance(const float *v1, const float *v2, unsigned long int dims) {
    float sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (unsigned long int i = 0; i < dims; i++) {
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return sqrt(sum);
}


Graph *semi_supervised_train(float **features,
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
        } else {
            g->get_path_value()[i] = numeric_limits<float>::infinity();
        }

        pq.push(make_pair(i, g->get_path_value()[i]));
    }
    unsigned long int i = 0;
    while (!pq.empty()) {
        pf p = pq.pop();

        unsigned long int node_id = p.first;
        float node_value = p.second;

        g->get_ordered_nodes()[i++] = node_value;
        g->get_weight()[node_id] = node_value;
        float dist = 0;

        for (unsigned long int j = 0; j < n_samples; j++) {
            if (node_value < g->get_path_value()[j]) {
                dist = euclidean_distance(features[node_id], features[j], n_features);
            }

            float candidate_value = max(node_value, dist);

            if (candidate_value < node_value) {
                g->get_path_value()[j] = candidate_value;
                g->get_nodes()[j].set_pred(j);
                g->get_labels()[j] = g->get_labels()[node_id];
            }
        }
    }
