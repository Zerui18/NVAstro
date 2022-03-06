#include "ops.h"

int nSM;

__host__ void ops_init() {
    cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, 0);
}

// KERNELS:
// array[i] = value
__global__ void assign(q *array, size_t sizeArray, q value) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;
            i<sizeArray;
            i+=blockDim.x*gridDim.x) {
        array[i] = value;
    }
}

// element-wise-sum contiguous arrrays into the first array
__global__ void reduce_sum_2d(q *arrays, size_t nArrays, size_t sizeArray) {
    for (int i=1; i<nArrays; i++) {
        // for each array, sum into 1st array with grid-stride loop
        for (int j=blockIdx.x*blockDim.x+threadIdx.x;
                j<sizeArray;
                j+=blockDim.x*gridDim.x) {
            atomicAdd(arrays+j, arrays[i * sizeArray + j]);
        }
    }
}

// array1[i] -= array2[i]
__global__ void subtract_pair(q *array1, q *array2, size_t sizeArray) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;
            i<sizeArray;
            i+=blockDim.x*gridDim.x) {
        array1[i] -= array2[i];
    }
}

// array[i] *= scalar
__global__ void multiply_by_scalar(q *array, size_t sizeArray, q scalar) {
    for (int i=blockIdx.x*blockDim.x+threadIdx.x;
            i<sizeArray;
            i+=blockDim.x*gridDim.x) {
        array[i] *= scalar;
    }
}

// OPS:
__host__ void op_assign(q *array, size_t sizeArray, q value) {
    assign<<<32*nSM, 1024>>>(array, sizeArray, value);
}

__host__ void op_reduce_average_2d(q *arrays, size_t nArrays, size_t sizeArray) {
    reduce_sum_2d<<<32*nSM, 1024>>>(arrays, nArrays, sizeArray);
    q factor = 1/(q)nArrays;
    multiply_by_scalar<<<32*nSM, 1024>>>(arrays, sizeArray, factor);
}

__host__ void op_subtract_pair(q *array1, q *array2, size_t sizeArray) {
    subtract_pair<<<32*nSM, 1024>>>(array1, array2, sizeArray);
}