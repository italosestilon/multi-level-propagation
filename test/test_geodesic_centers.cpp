#include <gtest/gtest.h>

#include <label_propagation/utils.h>

#include <unordered_map>

using namespace std;

TEST(GeodesicCenters, FindCenters) {
    uint32_t height = 20;
    uint32_t width = 20;
    uint32_t num_labels = 1;

    uint32_t *label_image = new uint32_t[height * width];

    for (uint32_t i = 0; i < height * width; i++) {
        label_image[i] = 0;
    }

    // create a square
    for (uint32_t i = 2; i < 12; i++) {
        for (uint32_t j = 2; j < 12; j++) {
            label_image[i * width + j] = 1;
        }
    }


    // compute square center
    unordered_map<uint32_t, pos> centers = geodesic_centers(label_image,
                                                height,
                                                width,
                                                num_labels);

    
    EXPECT_EQ(centers.size(), 1);
    EXPECT_EQ(centers[1].first, 6);
    EXPECT_EQ(centers[1].second, 6);
}