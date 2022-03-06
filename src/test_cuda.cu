#include <stdio.h>
#include <cassert>
#include <chrono>
#include "cuda/ops.h"

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %sn", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// HELPER
template <
    class result_t   = std::chrono::milliseconds,
    class clock_t    = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds
>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

auto start = std::chrono::steady_clock::now();

#define TIMER_START(label)\
start = std::chrono::steady_clock::now();\
printf(#label" [start]\n");

#define TIMER_END(label) printf(#label" [end]: %ldms\n", since(start).count());

#define IMG_WIDTH 5000
#define IMG_HEIGHT 4000
#define IMG_CHANNEL 3
#define IMG_SIZE IMG_WIDTH * IMG_HEIGHT * IMG_CHANNEL
#define N_IMG 20

#define float q;


void quickselect(q *array, size_t sizeArray, size_t lower, size_t upper, size_t k) {

}

int main(int argc, char **argv) {
    printf("Allocating imageArrays...\n");
    q *imageArrays;
    checkCuda(cudaMallocManaged(&imageArrays, N_IMG * IMG_SIZE * sizeof(q)));
    TIMER_START(assign)
    op_assign(imageArrays, N_IMG * IMG_SIZE, 256.0f);
    cudaDeviceSynchronize();
    TIMER_END(assign)
    printf("Testing...\n");
    TIMER_START(average)
    op_reduce_average_2d(imageArrays, N_IMG, IMG_SIZE);
    cudaDeviceSynchronize();
    TIMER_END(average)
    TIMER_START(minus)
    op_subtract_pair(imageArrays, imageArrays, IMG_SIZE);
    cudaDeviceSynchronize();
    TIMER_END(minus)
    printf("Test completed\n");
    printf("imageArrays[2] = %f\n", imageArrays[2]);
}