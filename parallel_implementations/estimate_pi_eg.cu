#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


__global__ void monteCarloPiEstimate(int* d_insideCircle, long long int n, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    int local_count = 0;

    for (int i = idx; i < n; i += stride) {
        float x = curand_uniform(&state) * 2.0f - 1.0f;
        float y = curand_uniform(&state) * 2.0f - 1.0f;
        float distance_squared = x * x + y * y;
        if (distance_squared <= 1)
            local_count++;
    }
        atomicAdd(d_insideCircle, local_count);
}


int main() {
    int n = 1 << 24;
    int h_insideCircle = 0;
    int* d_insideCircle;

    cudaMalloc(&d_insideCircle, sizeof(int));

    cudaMemcpy(d_insideCircle, &h_insideCircle, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;
    monteCarloPiEstimate << <numBlocks, blockSize >> > (d_insideCircle, n, time(NULL));

    cudaMemcpy(&h_insideCircle, d_insideCircle, sizeof(int), cudaMemcpyDeviceToHost);

    float piEstimate = 4.0f * h_insideCircle / n;
    printf("Estimated Pi = %f\n", piEstimate);

    cudaFree(d_insideCircle);

    return 0;
}