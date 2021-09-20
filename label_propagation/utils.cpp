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

double euclidean_distance(const float *v1, const float *v2, uint64_t dims) {
    float *diff = new float[dims];
    cblas_scopy(dims, v1, 1, diff, 1);

    // compute difference with BLAS
    cblas_saxpy(dims, -1.0f, v2, 1, diff, 1);

    // compute enclidean norm of difference
    double norm = cblas_snrm2(dims, diff, 1);

    delete[] diff;

    return norm;

}