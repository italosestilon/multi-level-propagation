#include <label_propagation/utils.h>
#include <label_propagation/priority_queue.h>
#include <limits>

// function to compute neihborhood of a pixel
vector<uint64_t> neighborhood(uint64_t pixel, uint64_t height, uint64_t width, uint32_t rad) {
    // unravel pixel index
    uint64_t x = pixel / width;
    uint64_t y = pixel % width;

    int32_t start = -(rad / 2);
    int32_t end = rad / 2;

    // compute neihborhood
    vector<uint64_t> neighbors;
    for (int64_t i = start; i <= end; i++) {
        for (int64_t j = start; j <= end; j++) {
            if (i == 0 && j == 0) continue;
            // check if neihbor is in image
            if (x + i < width && y + j < height) {
                // add neihbor to neihborhood
                uint64_t neighbor = (x + i) * width + (y + j);
                neighbors.push_back(neighbor);
            }
        }
    }

    return neighbors;
}

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

// function to unravel index
inline pair<uint32_t, uint32_t> unravel_index(uint64_t index, uint32_t height, uint32_t width) {
    uint32_t x = index / width;
    uint32_t y = index % width;
    return make_pair(x, y);
}


unordered_map<uint32_t, pos> geodesic_centers(uint32_t *labels,
                                              uint32_t height,
                                              uint32_t width,
                                              uint32_t num_labels) {
    
    // create map to store centers
    unordered_map<uint32_t, pos> centers;

    // create priority queue
    PairPQ pq(height * width);

    float *distance = new float[height * width];
    uint64_t *root = new uint64_t[height * width];

    for (uint64_t i = 0; i < height * width; i++) {
        distance[i] = numeric_limits<float>::max();
        if (labels[i] != 0) {
            vector<uint64_t> neighbors = neighborhood(i, height, width, 2.0);
            if (neighbors.size() < 8) {
                root[i] = i;
                distance[i] = 0.0;
                pq.push(make_pair(distance[i], i));
                
            } else {
                for (uint64_t j = 0; j < neighbors.size(); j++) {
                    uint64_t neighbor = neighbors[j];
                    uint32_t label = labels[neighbor];
                    if (label != labels[i]) {
                        distance[i] = 0.0;
                        root[i] = i;
                        pq.push(make_pair(i, 0.0));
                        break;
                    }
                }
            }
        }
    }

    // compute geodesic centers
    while (!pq.empty()) {
        pif first = pq.pop();
        uint64_t pixel = first.first;
        double dist = first.second;

        auto neighbors = neighborhood(pixel, height, width, 2.0);

        for(uint64_t i = 0; i < neighbors.size(); i++) {
            uint64_t neighbor = neighbors[i];
            if (labels[neighbor] != 0) {
                if (distance[neighbor] > dist && labels[neighbor] == labels[pixel]) {
                    // unravel root pixel index
                    int64_t p_x = root[pixel] / width;
                    int64_t p_y = root[pixel] % width;

                    // unravel neighbor index
                    int64_t n_x = neighbor / width;
                    int64_t n_y = neighbor % width;

                    // compute distance with double precision
                    double d = sqrt((p_x - n_x) * (p_x - n_x) + (p_y - n_y) * (p_y - n_y));

                    if (d < distance[neighbor]) {
                        if (distance[neighbor] != numeric_limits<float>::max()) {
                            // decrease key
                            uint64_t index = pq.get_index(neighbor);
                            pq.update(index, d);
                        } else {
                            // insert key
                            pq.push(make_pair(neighbor, d));
                        }

                        distance[neighbor] = d;
                        root[neighbor] = root[pixel];
                    }

                }
            }
        }
    }

    // compute geodesic centers
    for (uint64_t i = 0; i < height * width; i++) {
        uint32_t label = labels[i];
        if (label != 0) {
            if (centers.find(label) == centers.end()) {
                centers[label] = unravel_index(i, height, width);
            } else {
                pos center = centers[label];
                uint64_t center_index = center.first * width + center.second;

                if (distance[i] > distance[center_index]) {
                    centers[label] = unravel_index(i, height, width);
                }
            }
        }
    }

    return centers;
}