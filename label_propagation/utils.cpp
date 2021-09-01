#include <label_propagation/utils.h>

// generate linspaced arrays
std::vector<float> linspace(float start, float end, size_t N) {
    // create vector
    std::vector<float> linspaced_vector(N);

    float delta = (start + end) / static_cast<float>(N);

    // fill vector with linspaced values
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        linspaced_vector[i] = start + delta * static_cast<float>(i);
    }

    return linspaced_vector;

}