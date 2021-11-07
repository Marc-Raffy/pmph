#include <iostream>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <cub/cub.cuh>  
#include "kernel.cuh"
#include <chrono>

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
    for (int shif_elems = 16; shif_elems < 29; shif_elems++)
    {
        unsigned int num_elems = (1 << shif_elems);
        unsigned int* h_in = new unsigned int[num_elems];
        unsigned int* h_in_rand = new unsigned int[num_elems];
        unsigned int* h_out_gpu = new unsigned int[num_elems];
        unsigned int* h_out_cpu = new unsigned int[num_elems];
        for (int j = 0; j < num_elems; j++)
        {
            h_in[j] = (num_elems - 1) - j;
            h_in_rand[j] = rand() % num_elems;
        }

        

        std::chrono::steady_clock::time_point begin_cpu = std::chrono::steady_clock::now();
        cpu_sort(h_out_cpu, h_in_rand, num_elems);  
        std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();
        std::cout << "CPU runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - begin_cpu).count() << "[ms]" << std::endl;

        

        unsigned int* d_in;
        unsigned int* d_out;
        cudaMalloc(&d_in, sizeof(unsigned int) * num_elems);
        cudaMalloc(&d_out, sizeof(unsigned int) * num_elems);
        cudaMemcpy(d_in, h_in_rand, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice);

        std::chrono::steady_clock::time_point begin_gpu = std::chrono::steady_clock::now();
        radix_sort(d_out, d_in, num_elems);
        std::chrono::steady_clock::time_point end_gpu = std::chrono::steady_clock::now();
        std::cout << "GPU runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - begin_gpu).count() << "[ms]" << std::endl;

        bool match = true;
        cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_elems; ++i)
        {
            if (h_out_cpu[i] != h_out_gpu[i])
            {
                match = false;
            }
        }

        /*std::chrono::steady_clock::time_point begin_cub = std::chrono::steady_clock::now();
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_elems);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run sorting operation
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_out, num_elems);
        std::chrono::steady_clock::time_point end_cub = std::chrono::steady_clock::now();
        std::cout << "GPU runtime = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_cub - begin_cub).count() << "[ms]" << std::endl;
*/

        std::cout << "Match: " << match << std::endl;
        
        cudaMemcpy(h_out_gpu, d_out, sizeof(unsigned int) * num_elems, cudaMemcpyDeviceToHost);
        cudaFree(d_out);
        cudaFree(d_in);
    }
}