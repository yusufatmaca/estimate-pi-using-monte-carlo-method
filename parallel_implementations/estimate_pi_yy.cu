#include <curand_kernel.h>
#include <stdio.h>

__global__ void generate_random_numbers(unsigned long long int *count, unsigned long seed, unsigned long long int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    curand_init(seed + tid, tid, 0, &state);

    unsigned long long int local_counter = 0;

    for (int i = 0; i < stride; i++) {
      float x = curand_uniform(&state) * 2.0f - 1.0f;
      float y = curand_uniform(&state) * 2.0f - 1.0f;

      float distance_squared = x*x + y*y;

      if (distance_squared <= 1) local_counter++;
    }

    atomicAdd(count, local_counter);
}

int main() {
    // Device props
    // CC 7.5
    // 16 real block
    // 64 thread per block == 1024 maximum threads per multiprocessor
    // warp is 32

    int block = 32;
    int thread = 256;
    unsigned long long int count = 0, *cuda_count, n = 1e12;
    unsigned long long int stride = n / (block * thread);
    unsigned long seed = time(NULL);

    cudaMalloc((void**)&cuda_count, sizeof(unsigned long long int));
    cudaMemcpy(cuda_count, &count, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    generate_random_numbers<<<block, thread>>>(cuda_count, seed, stride);
    cudaMemcpy(&count, cuda_count, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double pi = 4 * count / ((double)n);
    printf("%0.12f\n", pi);

    return 0;
}

// RESULT -> 3.141593035940