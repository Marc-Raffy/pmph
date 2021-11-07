#include <iostream>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include "kernel.cuh"
#include<chrono>

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
    
    unsigned int num_elems = (1 << 29);
    unsigned int* h_in = new unsigned int[num_elems];
    unsigned int* h_in_rand = new unsigned int[num_elems];
    unsigned int* h_out_gpu = new unsigned int[num_elems];
    unsigned int* h_out_cub = new unsigned int[num_elems];
    unsigned int* h_out_cpu = new unsigned int[num_elems];
    /*for (int j = 0; j < num_elems; j++)
    {
        h_in[j] = (num_elems - 1) - j;
        h_in_rand[j] = rand() % num_elems;
    }

    cpu_sort(h_out_cpu, h_in_rand, num_elems);  
    double cpu_duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    std::cout << "CPU time: " << cpu_duration << " s" << std::endl;*/

    

    unsigned int* d_in;
    unsigned int* d_out;
    unsigned int* d_out_cub;
    cudaMalloc(&d_in, sizeof(unsigned int) * num_elems);
    cudaMalloc(&d_out, sizeof(unsigned int) * num_elems);
    cudaMalloc(&d_out_cub, sizeof(unsigned int) * num_elems);
    cudaMemcpy(d_in, h_in_rand, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    radix_sort(d_out, d_in, num_elems);
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    std::cout <<  microseconds << std::endl;
    
    cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost);

    
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out_cub, num_elems);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    auto start_cub = std::chrono::high_resolution_clock::now();
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out_cub, num_elems);
    auto elapsed_cub = std::chrono::high_resolution_clock::now() - start;
    long long microseconds_cub = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_cub).count();
    std::cout <<  microseconds_cub << std::endl;

    cudaMemcpy(h_out_cub, d_out_cub, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost);

    bool match = true;


    for (int i = 0; i < num_elems; ++i)
    {
        if (h_out_cub[i] != h_out_gpu[i])
        {
            match = false;
        }
    }
    std::cout << "Match: " << match << std::endl;
    /*for (int i = 0; i < num_elems; i++)
    {
        std::cout <<  h_out_gpu[i] << std::endl;
    }*/
    

    //cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_in);

}