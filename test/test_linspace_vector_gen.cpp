#include <cmath>
#include <label_propagation/utils.h>
#include <gtest/gtest.h>

// test if array is linspaced
TEST(Util, linspace_vector_gen) {
    float start = 0;
    float end = 2 * M_PI;
    unsigned int N = 5000;

    float step = (start + end) / static_cast<float>(N);

    std::vector<float> x = linspace(start, end, N);

    for (unsigned int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(x[i], start + i * step);
    }

    x.clear();
}
