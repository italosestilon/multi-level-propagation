#include <cmath>
#include <cstring>
#include <label_propagation/opf.h>
#include <label_propagation/utils.h>

#include <gtest/gtest.h>

#include <vector>

using namespace std;

TEST(OPF, Training) {
    float start = 0;
    float end = 2 * M_PI;
    unsigned int N = 500;

    const vector<float> linspaced_v = linspace(start, end, N);

    // outter circle
    vector<float> x_outer(N);
    vector<float> y_outer(N);
    for (unsigned int i = 0; i < N; i++) {
        x_outer[i] = cos(linspaced_v[i]);
        y_outer[i] = sin(linspaced_v[i]);
    }

    // inner circle
    vector<float> x_inner(N);
    vector<float> y_inner(N);
    for (unsigned int i = 0; i < N; i++) {
        x_inner[i] = cos(linspaced_v[i]) * 0.5;
        y_inner[i] = sin(linspaced_v[i]) * 0.5;
    }

    // labels
    unsigned int *labels = new unsigned int[2*N];
    for (unsigned int i = 0; i < 2*N; i++) {
        labels[i] = i < N ? 0 : 1;
    }

    // features
    float **features = new float*[2*N];

    for (unsigned int i = 0; i < N; i++) {
        features[i] = new float[2];
        features[i][0] = x_outer[i];
        features[i][0] = y_outer[i];
    }

    for (unsigned int i = 0; i < N; i++) {
        features[i + N] = new float[2];
        features[i + N][0] = x_inner[i];
        features[i + N][0] = y_inner[i];
    }

    // is supervised
    bool *is_supervised = new bool[2*N];
    for (unsigned int i = 0; i < 2*N; i++) {
        is_supervised[i] = true;
    }

    // train
    unsigned int n_samples = 2*N;
    unsigned int n_features_per_sample = 2;

    Graph *g = semi_supervised_train(features, labels, is_supervised, n_samples, n_features_per_sample);

    // test

}