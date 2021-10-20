#include <label_propagation/ift.h>
#include <label_propagation/utils.h>

#include <limits>

#include <random>

#include <chrono>

using namespace std::chrono;

void populate(float *features, uint64_t *seeds, float *opf_certainty, uint64_t num_pixels, uint64_t channels)
{
    mt19937 rng;
    uniform_real_distribution<float> features_dist(0.0, 100.0);
    normal_distribution<float> noise_dist(0.0, 1.0);
    uniform_real_distribution<float> certainty_dist(0.5, 1.0);
    bernoulli_distribution seeds_dist(0.05);
    bernoulli_distribution labels_dist(0.25);

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
}

int main()
{
    uint64_t height = 1024;
    uint64_t width = 1024;
    uint64_t channels = 64;
    uint64_t num_pixels = height * width;

    float *features = new float[num_pixels * channels];
    uint64_t *seeds = new uint64_t[num_pixels];
    float *opf_certainty = new float[num_pixels];

    populate(features, seeds, opf_certainty, num_pixels, channels);

    uint64_t *pred_out = new uint64_t[num_pixels];
    uint64_t *root_out = new uint64_t[num_pixels];
    double *cost_out = new double[num_pixels];

    printf("Computing IFT\n");

    auto start = high_resolution_clock::now();
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

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(end - start);
    printf("IFT took %f seconds\n", (double)duration.count());

    printf("Computing certainty\n");

    start = high_resolution_clock::now();
    double *certainty = compute_certainty(
        height,
        width,
        cost_out,
        labels_from_ift,
        root_out,
        features,
        channels,
        3);
    end = high_resolution_clock::now();
    duration = duration_cast<seconds>(end - start);
    printf("Certainty took %f seconds\n", (double)duration.count());

    delete[] certainty;
    delete[] pred_out;
    delete[] root_out;
    delete[] cost_out;
    delete[] features;
    delete[] opf_certainty;
    delete[] seeds;
    delete[] labels_from_ift;
    return 0;
}
