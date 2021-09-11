#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void squareKernel(float* d_in, float *d_out) {
    const unsigned int lid = threadIdx.x;
    const unsigned int gid = blockIdx.x*blockDim.x + lid;
    d_out[gid] = d_in[gid]*d_in[gid];
}

void cpu_function(float* array_input, float* array_output, int array_size){
	for (int i = 0; i < array_size; i++)
	{
		array_output[i] = pow((array_input[i]/array_input[i]-2.3), 3);
	}
}

int main(int argc, char** argv){
	unsigned int N = 512;
    unsigned int mem_size = N*sizeof(float);

    // allocate host memory for GPU function
    float* h_in  = (float*) malloc(mem_size);
    float* h_out = (float*) malloc(mem_size);
	//allocate host memory for CPU function
	//float* array_input  = (float*) malloc(mem_size);
    //float* array_output = (float*) malloc(mem_size);

    // initialize the memory
    for(unsigned int i=1; i<=N; ++i) {
        h_in[i] = (float)i;
		//array_input[i] = (float)i;
    }
    //allocate device memory
	float* d_in;
    float* d_out;
    cudaMalloc((void**)&d_in,  mem_size);
    cudaMalloc((void**)&d_out, mem_size);

    cudaMemcpy(d_in, h_in, mem_size, cudaMemcpyHostToDevice);

    squareKernel<<< 1, 128>>>(d_in, d_out);

    cudaMemcpy(h_out, d_out, mem_size, cudaMemcpyDeviceToHost);

    // print result
    for(unsigned int i=0; i<N; ++i) printf("%.6f\n", h_out[i]);

    //free(array_input); //free(array_output);
    free(h_in);        free(h_out);
    cudaFree(d_in);    cudaFree(d_out);

}