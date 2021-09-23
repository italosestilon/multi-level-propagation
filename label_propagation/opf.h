# pragma once

#include <cstdint>
#include <limits>

using namespace std;

// define a node in the graph
class Node {
public:
    // default constructor
    Node() = default;
    // constructor
    Node(float *features, unsigned long int n_features) {
        features = features;
        n_features = n_features;
    }

    // getters and setters
    float *get_features() {
        return features;
    }
    unsigned long int get_n_features() {
        return n_features;
    }
    void set_features(float *features) {
        this->features = features;
    }
    void set_n_features(unsigned long int n_features) {
        this->n_features = n_features;
    }

    long unsigned int get_pred() {
        return pred;
    }

    void set_pred(long unsigned int pred) {
        this->pred = pred;
    }

    // destructor
    ~Node() {}
private:
    float *features;
    unsigned long int n_features;
    long unsigned int pred;
};

// define class that represents a graph
class Graph
{
public:
    // constructor
    Graph(long unsigned int n_nodes,
            long unsigned int n_features,
            float *features) {
        
        this->n_nodes = n_nodes;
        this->n_features = n_features;
        this->nodes = new Node[n_nodes];
        this->path_value = new double[n_nodes];
        this->weight = new double[n_nodes];
        this->labels = new unsigned int[n_nodes];
        this->ordered_nodes = new long unsigned int[n_nodes];
        this->supervised = new bool[n_nodes];

        #pragma omp parallel for
        for (long unsigned int i = 0; i < n_nodes; i++) {
            this->nodes[i].set_features(features + i);
            this->nodes[i].set_n_features(n_features);
            this->path_value[i] = numeric_limits<float>::infinity();
        }
    }

    // getters
    long unsigned int get_n_nodes() {
        return n_nodes;
    }
    long unsigned int get_n_features() {
        return n_features;
    }
    Node *get_nodes() {
        return nodes;
    }
    double *get_path_value() {
        return path_value;
    }
    long unsigned int *get_ordered_nodes() {
        return ordered_nodes;
    }

    unsigned int *get_labels() {
        return labels;
    }

    bool *is_supervised() {
        return supervised;
    }

    double *get_weight() {
        return weight;
    }

    float* get_node_features(long unsigned int node_id) {
        return nodes[node_id].get_features();
    }

    // destructor
    ~Graph() {
        delete[] nodes;
        delete[] path_value;
        delete[] ordered_nodes;
        delete[] labels;
    }

private:
    unsigned long int n_nodes;
    unsigned long int n_features;
    Node *nodes;
    unsigned int *labels;
    double *path_value;
    double *weight;
    long unsigned int *ordered_nodes;
    bool *supervised;
};

Graph *semi_supervised_train(float *features,
                             unsigned int *labels,
                             bool *is_supervised,
                             unsigned long int n_samples,
                             unsigned long int n_features);

void classify_with_certainty(Graph *graph,
                             float *features,
                             uint64_t n_samples,
                             uint64_t n_features,
                             uint32_t *labels_out,
                             double *certainties_out);