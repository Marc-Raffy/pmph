#include <iostream>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include "kernel.cuh"

void cpu_sort(unsigned int* h_out, unsigned int* h_in, size_t len)
{
    for (int i = 0; i < len; ++i)
    {
        h_out[i] = h_in[i];
    }
    std::sort(h_out, h_out + len);
}

int main()
{
    std::clock_t start;
    
    unsigned int num_elems = (1 << 13);
    unsigned int* h_in = new unsigned int[num_elems];
    unsigned int* h_in_rand = new unsigned int[num_elems];
    unsigned int* h_out_gpu = new unsigned int[num_elems];
    unsigned int* h_out_cpu = new unsigned int[num_elems];
    for (int j = 0; j < num_elems; j++)
    {
        h_in[j] = (num_elems - 1) - j;
        h_in_rand[j] = rand() % 1024;
    }

    cpu_sort(h_out_cpu, h_in_rand, num_elems);

    unsigned int* d_in;
    unsigned int* d_out;
    cudaMalloc(&d_in, sizeof(unsigned int) * num_elems);
    cudaMalloc(&d_out, sizeof(unsigned int) * num_elems);
    cudaMemcpy(d_in, h_in_rand, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice);

    start = std::clock();
    radix_sort(d_out, d_in, num_elems);
    double gpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    
    std::cout << "GPU time: " << gpu_duration << " s" << std::endl;
    bool match = true;
    int index_diff = 0;
    cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_elems; ++i)
    {
        std::cout<< h_out_cpu[i] << " -- " << h_out_gpu[i]<< std::endl;
        if (h_out_cpu[i] != h_out_gpu[i])
        {
            
            match = false;
            index_diff = i;
            break;
        }
    }
    std::cout << "Match: " << match << std::endl;
    /*for (int i = 0; i < num_elems; i++)
    {
        std::cout <<  h_out_gpu[i] << std::endl;
    }*/
    

    cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_in);

}