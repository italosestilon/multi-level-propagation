# pragma once

#include <limits>

#include <label_propagation/priority_queue.h>

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
            float **features) {

        this->n_nodes = n_nodes;
        this->n_features = n_features;
        this->nodes = new Node[n_nodes];
        this->path_value = new float[n_nodes];
        this->labels = new unsigned int[n_nodes];
        this->weight = new float[n_nodes];

        #pragma omp parallel for
        for (long unsigned int i = 0; i < n_nodes; i++) {
            this->nodes[i].set_features(features[i]);
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
    float *get_path_value() {
        return path_value;
    }
    long unsigned int *get_ordered_nodes() {
        return ordered_nodes;
    }

    unsigned int *get_labels() {
        return labels;
    }

    float *get_weight() {
        return weight;
    }

    // destructor
    ~Graph() {
        delete[] nodes;
        delete[] path_value;
        delete[] ordered_nodes;
        delete[] labels;
        delete[] weight;
    }

private:
    unsigned long int n_nodes;
    unsigned long int n_features;
    Node *nodes;
    unsigned int *labels;
    float *path_value;
    float *weight;
    long unsigned int *ordered_nodes;
};

Graph *semi_supervised_train(float **features,
                             unsigned int *labels,
                             bool *is_supervised,
                             unsigned long int n_samples,
                             unsigned long int n_features);
