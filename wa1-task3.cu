#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void squareKernel(float* d_in, float *d_out, int N) {
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + lid;
    if(gid < N)
    {
        d_out[gid] = pow(d_in[gid]/(d_in[gid]-2.3),3);
    }
}

void cpu_function(float* array_input, float* array_output, int array_size){
	for (int i = 0; i < array_size; i++)
	{
		array_output[i] = pow((array_input[i]/array_input[i]-2.3), 3);
	}
}

int main(int argc, char** argv){
	unsigned int N = 753412;
    unsigned int mem_size = N*sizeof(float);
    unsigned int block_size = 256;
    unsigned int num_blocks = ((N + (block_size - 1) / block_size));
    
    // allocate host memory for GPU function
    float* h_in  = (float*) malloc(mem_size);
    float* h_out = (float*) malloc(mem_size);
	//allocate host memory for CPU function
	float* array_input  = (float*) malloc(mem_size);
    float* array_output = (float*) malloc(mem_size);
    // initialize the memory
    for(unsigned int i=1; i<N; ++i) {
        h_in[i] = (float)i;
		array_input[i] = (float)i;
    }
    //runs CPU function
    cpu_function(array_input, array_output, N);

    //allocate device memory
	float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    squareKernel<<< num_blocks, block_size>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    // check if results match
    int flag = 0;
    for(unsigned int i=1; i<N; ++i) {
        if(array_output[i] != h_out[i]){
            flag++;
        }
    }
    printf("%f Number of elements that do not match", flag);

    free(array_input); free(array_output);
    free(h_in);        free(h_out);
    cudaFree(d_in);    cudaFree(d_out);

}