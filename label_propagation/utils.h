#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <utility>

#include <unordered_map>

extern "C"
{
   #include <cblas.h>
}


using namespace std;

typedef pair<uint32_t, uint32_t> pos;

vector<uint64_t> neighborhood(uint64_t pixel, uint64_t height, uint64_t width, uint32_t rad=3);

// a function to generate linspaced arrays
std::vector<float> linspace(float start, float end, size_t N);

double euclidean_distance(const float *v1, const float *v2, uint64_t dims);

unordered_map<uint32_t, pos> geodesic_centers(uint32_t *labels,
                                              uint32_t height,
                                              uint32_t width,
                                              uint32_t num_labels);