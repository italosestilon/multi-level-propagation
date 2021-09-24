#include <gtest/gtest.h>

#include <label_propagation/opf.h>
#include <label_propagation/utils.h>

#include <limits>

#include <vector>

#include <random>

using namespace std;

TEST(OPFTest, Train) {
    uint32_t n_points = 200;
    uint32_t n_features = 2;
    float scale = 3.0;

    vector<float> space = linspace(0, 1, n_points/2);

    float *features = new float[n_points * n_features];
    uint32_t *labels = new uint32_t[n_points];
    bool *is_supervised = new bool[n_points];

    for (uint32_t i = 0; i < n_points/2; i++) {
        features[i * n_features + 0] = sin(space[i]);
        features[i * n_features + 1] = cos(space[i]);
        labels[i] = 1;
    }

    for (uint32_t i = n_points/2, j = 0; i < n_points; i++, j++) {
        features[i * n_features + 0] = scale * sin(space[j]);
        features[i * n_features + 1] = scale * cos(space[j]);
        labels[i] = 2;
    }
    
    // populate is_supervised randomly
    mt19937 rng;
    uint32_t n_train = 0;
    bernoulli_distribution seeds_dist(0.6);
    for (uint32_t i = 0; i < n_points; i++) {
        is_supervised[i] = seeds_dist(rng);
        if (is_supervised[i]) {
            n_train++;
        }
    }

    // select training set
    float *train_features = new float[n_train * n_features];
    uint32_t *train_labels = new uint32_t[n_train];
    bool *train_is_supervised = new bool[n_train];

    for(uint32_t i = 0, j = 0; i < n_points; i++) {
        if (is_supervised[i]) {
            memcpy(train_features + j * n_features, features + i * n_features, n_features * sizeof(float));
            train_labels[j] = labels[i];
            train_is_supervised[j] = is_supervised[i];
            j++;
        }
    }

    // select test set
    uint32_t n_test = n_points - n_train;
    float *test_features = new float[n_test * n_features];
    uint32_t *test_labels = new uint32_t[n_test];

    for(uint32_t i = 0, j = 0; i < n_points; i++) {
        if (!is_supervised[i]) {
            memcpy(test_features + j * n_features, features + i * n_features, n_features * sizeof(float));
            test_labels[j] = labels[i];
            j++;
        }
    }

    // create the model
    Graph *graph = semi_supervised_train(train_features,
                                        train_labels,
                                        train_is_supervised,
                                        n_train,
                                        n_features);

    // classify the test set
    uint32_t *test_predictions = new uint32_t[n_test];
    double *test_certainty = new double[n_test];

    classify_with_certainty(graph,
                            test_features,
                            n_test,
                            n_features,
                            test_predictions,
                            test_certainty);

    
    // check that the predictions are correct
    uint64_t correct = 0;
    for (uint32_t i = 0; i < n_test; i++) {
       if (test_predictions[i] == test_labels[i]) {
            correct++;
        }
    }

    EXPECT_GT(correct / (double) n_test, 0.95);

    // check that the certainty is correct
    for (uint32_t i = 0; i < n_test; i++) {
        EXPECT_GE(test_certainty[i], 0.5);
        EXPECT_LE(test_certainty[i], 1.0);
    }
    
    delete graph;
    delete[] features;
    delete[] labels;
    delete[] is_supervised;
    delete[] train_features;
    delete[] train_labels;
    delete[] train_is_supervised;
    delete[] test_features;
    delete[] test_labels;
    delete[] test_predictions;
    delete[] test_certainty;
}