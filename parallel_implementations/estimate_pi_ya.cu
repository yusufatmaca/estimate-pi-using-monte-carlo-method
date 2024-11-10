#include <stdio.h>
#include <curand_kernel.h>

__global__
void estimate_pi(int *d_number_in_circle, int number_of_tosses, unsigned long seed)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  curandState state;
  curand_init(seed + index, index, 0, &state);

  int local_count = 0;

  for (int i = index; i < number_of_tosses; i += stride) {
        float x = curand_uniform(&state) * 2.0f - 1.0f;
        float y = curand_uniform(&state) * 2.0f - 1.0f;
        float distance_squared = x*x + y*y;
        if (distance_squared <= 1)
        {
            local_count++;
        }
    }
  atomicAdd(d_number_in_circle, local_count); // to prevent race conditions
}

int main(void)
{
  int number_of_tosses = 1 << 24;    // Total number of points (tosses)
  int block_size = 256;              // Number of threads per block
  int num_blocks = 256;              // Number of blocks

  int *d_number_in_circle;
  cudaMalloc(&d_number_in_circle, sizeof(int));
  cudaMemset(d_number_in_circle, 0, sizeof(int));

  unsigned long seed = time(NULL);

  estimate_pi<<<num_blocks, block_size>>>(d_number_in_circle, number_of_tosses, seed);

  int number_in_circle;
  cudaMemcpy(&number_in_circle, d_number_in_circle, sizeof(int), cudaMemcpyDeviceToHost);

  float pi_estimated = 4.0f * number_in_circle / number_of_tosses;
  printf("Estimated Pi = %f\n", pi_estimated);

  cudaFree(d_number_in_circle);

  return 0;
}