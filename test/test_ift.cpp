#include <ift/core/io/NumPy.h>
#include <ift/core/dtypes/BasicDataTypes.h>

#include <gtest/gtest.h>

#include <label_propagation/ift.h>

#include <limits>

#include <vector>

using namespace std;

TEST(IFT, load_array) {
    long *shape = new long[3];
    shape[0] = 1024;
    shape[1] = 1024;
    shape[2] = 64;
    size_t n_dims = 3;

    ift_numpy_header *features_header = iftCreateNumPyHeader(
        IFT_FLT_TYPE, 
        shape,
        n_dims);

    ift_numpy_header *seeds_header = iftCreateNumPyHeader(
        IFT_LONG_TYPE,
        shape,
        2);

    void *features_data = iftReadNumPy("../../features_ift.npy", &features_header);
    void* seeds_data = iftReadNumPy("../../labels_to_ift.npy", &seeds_header);

    //ASSERT_EQ(header->dtype, IFT_FLT_TYPE);
    //ASSERT_EQ(header->n_dims, n_dims);
    //ASSERT_EQ(header->shape[0], 1024);
    //ASSERT_EQ(header->shape[1], 1024);

    float *features = (float*)features_data;
    int64_t *seeds = (int64_t*)seeds_data;

    int64_t *new_seeds = new int64_t[1024*1024];
    for (uint64_t i = 0; i < 1024; i++) {
        for (uint64_t  j = 0; j < 1024; j++) {
            int64_t seed = seeds[i*1024 + j];
            new_seeds[i*1024 + j] = seed;
        }
    }

    iftWriteNumPy(seeds_header, new_seeds, "../../seeds_test.npy");

    delete[] shape;
    delete[] features_header;
    delete[] seeds_header;
    delete[] new_seeds;
}

TEST(IFT, WriteArray) {
    const char* features_filename = "../../test.npy";

    long *shape = new long[3];
    shape[0] = 1024;
    shape[1] = 1024;
    shape[2] = 64;
    size_t n_dims = 2;

    ift_numpy_header *features_header = iftCreateNumPyHeader(
        IFT_FLT_TYPE, 
        shape,
        n_dims);

    float *features = new float[1024*1024*64];

    iftWriteNumPy(features_header, features, features_filename);
}

TEST(IFT, neighbors) {
    vector<uint64_t> neighbors = neighborhood(0, 1024, 1024);
    ASSERT_EQ(neighbors.size(), 3);

    neighbors = neighborhood(5, 1024, 1024);
    ASSERT_EQ(neighbors.size(), 5);

    neighbors = neighborhood(6248, 1024, 1024);
    ASSERT_EQ(neighbors.size(), 8);
}

TEST(IFT, computeIFT) {
    long *shape = new long[3];
    shape[0] = 1024;
    shape[1] = 1024;
    shape[2] = 64;
    size_t n_dims = 3;

    ift_numpy_header *features_header = iftCreateNumPyHeader(
        IFT_FLT_TYPE, 
        shape,
        n_dims);

    ift_numpy_header *seeds_header = iftCreateNumPyHeader(
        IFT_LONG_TYPE,
        shape,
        2);

    void *features_data = iftReadNumPy("../../features_ift.npy", &features_header);
    void* seeds_data = iftReadNumPy("../../labels_to_ift.npy", &seeds_header);

    float *features = (float*)features_data;
    uint64_t *seeds = (uint64_t*)seeds_data;
    uint32_t height = shape[0];
    uint32_t width = shape[1];
    uint64_t n_features = shape[2];
    uint64_t n_nodes = height * width;

    uint64_t *pred_out = new uint64_t[n_nodes];
    uint64_t *root_out = new uint64_t[n_nodes];
    float *cost_out = new float[n_nodes];
    bool *visited_out = new bool[n_nodes];

    for (uint64_t i = 0; i < n_nodes; i++) {
        visited_out[i] = false;
    }

    compute_itf(features,
                height,
                width,
                seeds,
                n_nodes,
                n_features,
                pred_out,
                root_out,
                cost_out,
                visited_out);
    
    uint64_t *labels_from_ift = new uint64_t[n_nodes];
    for (uint64_t i = 0; i < n_nodes; i++) {
        ASSERT_TRUE(visited_out[i]);
        ASSERT_TRUE(cost_out[i] < numeric_limits<float>::max());
        if (seeds[i] != 0) {
            ASSERT_TRUE(cost_out[i] == 0.0);
            ASSERT_TRUE(pred_out[i] == i);
            ASSERT_TRUE(root_out[i] == i);
        } else {
            ASSERT_TRUE(pred_out[i] != i);
            ASSERT_TRUE(root_out[i] != i);
        }
        
        labels_from_ift[i] = seeds[root_out[i]];

    }

    float *certainty = compute_certainty(height, width, root_out, labels_from_ift);

    // test if certainty is greater than zero
    for (uint64_t i = 0; i < n_nodes; i++) {
        ASSERT_TRUE(certainty[i] > 0.0);
    }

    ift_numpy_header *certainty_header = iftCreateNumPyHeader(
        IFT_FLT_TYPE,
        shape,
        2);

    ift_numpy_header *pred_header = iftCreateNumPyHeader(
        IFT_ULONG_TYPE,
        shape,
        2);

    ift_numpy_header *root_header = iftCreateNumPyHeader(
        IFT_ULONG_TYPE,
        shape,
        2);

    ift_numpy_header *labels_header = iftCreateNumPyHeader(
        IFT_ULONG_TYPE,
        shape,
        2);
    

    iftWriteNumPy(certainty_header, certainty, "../../certainty_ift.npy");
    iftWriteNumPy(pred_header, pred_out, "../../pred_ift.npy");
    iftWriteNumPy(root_header, root_out, "../../root_ift.npy");
    iftWriteNumPy(labels_header, labels_from_ift, "../../labels_from_ift.npy");

    delete[] certainty;
    delete[] certainty_header;
    delete[] pred_header;
    delete[] root_header;
    delete[] pred_out;
    delete[] root_out;
    delete[] cost_out;
    delete[] visited_out;
    delete[] features_header;
    delete[] seeds_header;
    delete[] features;
    delete[] seeds;
    delete[] labels_from_ift;
    delete[] shape;
}

     