#include <iostream>
#include <iomanip>
#include <ctime>
#include "kernel.cu.h"
int main()
{
    std::clock_t start;
    
    unsigned int num_elems = (1 << 16);
    unsigned int* h_in = new unsigned int[num_elems];
    unsigned int* h_in_rand = new unsigned int[num_elems];
    unsigned int* h_out_gpu = new unsigned int[num_elems];

    for (int j = 0; j < num_elems; j++)
    {
        h_in[j] = (num_elems - 1) - j;
        h_in_rand[j] = rand() % num_elems;
    }

    unsigned int* d_in;
    unsigned int* d_out;
    cudaMalloc(&d_in, sizeof(unsigned int) * num_elems);
    cudaMalloc(&d_out, sizeof(unsigned int) * num_elems);
    cudaMemcpy(d_in, h_in_rand, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice);
    start = std::clock();
    radix_sort(d_out, d_in, num_elems);
    double gpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "GPU time: " << gpu_duration << " s" << std::endl;
    cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_in);

}