#include <gtest/gtest.h>

#include <label_propagation/ift.h>
#include <label_propagation/utils.h>

#include <limits>

#include <vector>

#include <random>

using namespace std;

TEST(IFT, euclidean_distance)
{
    // generate two random points as arrays
    const uint32_t size = 64;
    float *p1 = new float[size];
    float *p2 = new float[size];

    mt19937 rng;
    uniform_real_distribution<float> dist(-1.0, 1.0);
    // populate with random values
    for (uint32_t i = 0; i < size; i++)
    {
        p1[i] = dist(rng);
        p2[i] = dist(rng);
    }

    // compute distance
    double dist_euclidean = euclidean_distance(p1, p2, size);

    // compute distance manually
    double dist_manual = 0.0;
    for (uint32_t i = 0; i < size; i++)
    {
        dist_manual += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    dist_manual = sqrt(dist_manual);

    // check if the two distances are equal
    EXPECT_NEAR(dist_euclidean, dist_manual, 1e-6);

    delete[] p1;
    delete[] p2;
}

TEST(IFT, neighbors)
{
    vector<uint64_t> neighbors = neighborhood(0, 1024, 1024);
    ASSERT_EQ(neighbors.size(), 3);

    neighbors = neighborhood(5, 1024, 1024);
    ASSERT_EQ(neighbors.size(), 5);

    neighbors = neighborhood(6248, 1024, 1024);
    ASSERT_EQ(neighbors.size(), 8);
}

TEST(IFT, computeIFT)
{
    uint64_t height = 780;
    uint64_t width = 1316;
    uint64_t channels = 64;
    uint64_t num_pixels = height * width;

    mt19937 rng;
    uniform_real_distribution<float> features_dist(0.0, 100.0);
    normal_distribution<float> noise_dist(0.0, 1.0);
    uniform_real_distribution<float> certainty_dist(0.5, 1.0);
    bernoulli_distribution seeds_dist(0.05);
    bernoulli_distribution labels_dist(0.25);

    float *features = new float[num_pixels * channels];
    uint64_t *seeds = new uint64_t[num_pixels];
    float *opf_certainty = new float[num_pixels];

    // populate features
    for (uint64_t i = 0; i < num_pixels; i++)
    {
        for (uint64_t j = 0; j < channels; j++)
        {
            features[i * channels + j] = features_dist(rng) + noise_dist(rng);
        }
    }

    // populate seeds
    for (uint64_t i = 0; i < num_pixels; i++)
    {
        bool is_seed = seeds_dist(rng);
        seeds[i] = is_seed ? (uint64_t)labels_dist(rng) + 1 : 0;
    }

    // populate opf_certainty
    for (uint64_t i = 0; i < num_pixels; i++)
    {
        opf_certainty[i] = certainty_dist(rng);
    }

    uint64_t *pred_out = new uint64_t[num_pixels];
    uint64_t *root_out = new uint64_t[num_pixels];
    double *cost_out = new double[num_pixels];

    compute_ift(features,
                height,
                width,
                seeds,
                opf_certainty,
                num_pixels,
                channels,
                3,
                pred_out,
                root_out,
                cost_out);

    uint64_t *labels_from_ift = new uint64_t[num_pixels];
    for (uint64_t i = 0; i < num_pixels; i++)
    {
        ASSERT_TRUE(cost_out[i] < numeric_limits<double>::max());
        /*if (seeds[i] != 0) {
            ASSERT_TRUE(cost_out[i] == 0.0);
            ASSERT_TRUE(pred_out[i] == i);
            ASSERT_TRUE(root_out[i] == i);
        } else {
            ASSERT_TRUE(pred_out[i] != i);
            ASSERT_TRUE(root_out[i] != i);
        }*/

        labels_from_ift[i] = seeds[root_out[i]];
    }

    double *certainty = compute_certainty(
        height,
        width,
        cost_out,
        labels_from_ift,
        root_out,
        features,
        channels,
        3);

    // test if certainty is greater than zero
    for (uint64_t i = 0; i < num_pixels; i++)
    {
        ASSERT_TRUE(certainty[i] >= 0.5);
    }

    delete[] certainty;
    delete[] pred_out;
    delete[] root_out;
    delete[] cost_out;
    delete[] features;
    delete[] opf_certainty;
    delete[] seeds;
    delete[] labels_from_ift;
}
