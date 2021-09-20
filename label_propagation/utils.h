#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>
extern "C"
{
   #include <cblas.h>
}
// a function to generate linspaced arrays
std::vector<float> linspace(float start, float end, size_t N);

double euclidean_distance(const float *v1, const float *v2, uint64_t dims);