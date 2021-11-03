#define BLOCK_SIZE 128
#include <cub/cub.cuh>
#include <iostream>
//NVIDIA prefix sum scan

__global__ void gpu_radix_sort_local(unsigned int* d_out_sorted,
    unsigned int* d_prefix_sums,
    unsigned int* d_block_sums,
    unsigned int input_shift_width,
    unsigned int* d_in,
    unsigned int d_in_len,
    unsigned int max_elems_per_block)
{
    extern __shared__ unsigned int shmem[];
    unsigned int* s_data = shmem;
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int* s_mask_out = &s_data[max_elems_per_block];
    unsigned int* s_merged_scan_mask_out = &s_mask_out[s_mask_out_len];
    unsigned int* s_mask_out_sums = &s_merged_scan_mask_out[max_elems_per_block];
    unsigned int* s_scan_mask_out_sums = &s_mask_out_sums[4];
    unsigned int thIdx = threadIdx.x;
    //cpy_idx is the global index of the current thread
    unsigned int cpy_idx = BLOCK_SIZE * blockIdx.x + thIdx;
    //Check that we currently within the bounds of the input array
    if (cpy_idx < d_in_len)
        s_data[thIdx] = d_in[cpy_idx];
    else
        s_data[thIdx] = 0;
    //Synchronize threads so that shared data is properly innitialized
    __syncthreads();
    unsigned int t_data = s_data[thIdx];
    unsigned int radix = (s_data[thIdx] >> input_shift_width) & 3;
    //s_mask_out = &s_data[128];
    for (unsigned int i = 0; i < 4; ++i)
    {
        //Need explanations 
        // Zero out s_mask_out
        s_mask_out[thIdx] = 0;
        if (thIdx == 0)
            s_mask_out[s_mask_out_len - 1] = 0;
        __syncthreads();

        // build bit mask output
        bool val_equals_i = false;
        if (cpy_idx < d_in_len)
        {
            val_equals_i = radix == i;
            s_mask_out[thIdx] = val_equals_i;
        }
        __syncthreads();
        //Hillis & Steele Parallel Scan Algorithm
        unsigned int sum = 0;
        unsigned int max_steps = (unsigned int) log2f(max_elems_per_block);
        for (unsigned int d = 0; d < max_steps; d++) {
            if (thIdx < 1 << d) {
                sum = s_mask_out[thIdx];
            }
            else {
                sum = s_mask_out[thIdx] + s_mask_out[thIdx - (1 << d)];
                
            }
            __syncthreads();
            s_mask_out[thIdx] = sum;
            __syncthreads();
        }
        //Turn inclusive to exclusive scan
        unsigned int cpy_val = 0;
        cpy_val = s_mask_out[thIdx];
        __syncthreads();
        s_mask_out[thIdx + 1] = cpy_val;
        __syncthreads();

        if (thIdx == 0)
        {
            // Zero out first element to produce the same effect as exclusive scan
            s_mask_out[0] = 0;
            unsigned int total_sum = s_mask_out[s_mask_out_len - 1];
            s_mask_out_sums[i] = total_sum;
            d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
        }
        __syncthreads();
        if (val_equals_i && (cpy_idx < d_in_len))
        {
            s_merged_scan_mask_out[thIdx] = s_mask_out[thIdx];
        }
        __syncthreads();
    }  

    // Scan mask output sums
    // Just do a naive scan since the array is really small
    if (thIdx == 0)
    {
        unsigned int run_sum = 0;
        for (unsigned int i = 0; i < 4; ++i)
        {
            s_scan_mask_out_sums[i] = run_sum;
            run_sum += s_mask_out_sums[i];
        }
    }

    __syncthreads();

    if (cpy_idx < d_in_len)
    {
        // Calculate the new indices of the input elements for sorting
        unsigned int t_prefix_sum = s_merged_scan_mask_out[thIdx];
        unsigned int new_pos = t_prefix_sum + s_scan_mask_out_sums[radix];
        
        __syncthreads();

        // Shuffle the block's input elements to actually sort them
        // Do this step for greater global memory transfer coalescing
        //  in next step
        s_data[new_pos] = t_data;
        s_merged_scan_mask_out[new_pos] = t_prefix_sum;
        
        __syncthreads();

        // Copy block - wise prefix sum results to global memory
        // Copy block-wise sort results to global 
        d_prefix_sums[cpy_idx] = s_merged_scan_mask_out[thIdx];
        d_out_sorted[cpy_idx] = s_data[thIdx];
    }
}

__global__ void gpu_glbl_shuffle(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_scan_block_sums,
    unsigned int* d_prefix_sums,
    unsigned int input_shift_width,
    unsigned int d_in_len,
    unsigned int max_elems_per_block)
{
    // get d = digit
    // get n = blockIdx
    // get m = local prefix sum array value
    // calculate global position = P_d[n] + m
    // copy input element to final position in d_out

    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = max_elems_per_block * blockIdx.x + thid;

    if (cpy_idx < d_in_len)
    {
        unsigned int t_data = d_in[cpy_idx];
        unsigned int t_2bit_extract = (t_data >> input_shift_width) & 3;
        unsigned int t_prefix_sum = d_prefix_sums[cpy_idx];
        unsigned int data_glbl_pos = d_scan_block_sums[t_2bit_extract * gridDim.x + blockIdx.x]
            + t_prefix_sum;
        __syncthreads();
        d_out[data_glbl_pos] = t_data;
    }
}

void radix_sort(unsigned int* const d_out,
    unsigned int* const d_in,
    unsigned int d_in_len)
{
    unsigned int block_sz = BLOCK_SIZE;
    unsigned int max_elems_per_block = block_sz;
    unsigned int grid_sz = d_in_len / max_elems_per_block;
    // Take advantage of the fact that integer division drops the decimals
    if (d_in_len % max_elems_per_block != 0)
        grid_sz += 1;

    unsigned int* d_prefix_sums;
    unsigned int d_prefix_sums_len = d_in_len;
    cudaMalloc(&d_prefix_sums, sizeof(unsigned int) * d_prefix_sums_len);
    cudaMemset(d_prefix_sums, 0, sizeof(unsigned int) * d_prefix_sums_len);

    unsigned int* d_block_sums;
    unsigned int d_block_sums_len = 4 * grid_sz; // 4-way split
    cudaMalloc(&d_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(d_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    unsigned int* d_scan_block_sums;
    cudaMalloc(&d_scan_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(d_scan_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    // shared memory consists of 3 arrays the size of the block-wise input
    //  and 2 arrays the size of n in the current n-way split (4)
    unsigned int s_data_len = max_elems_per_block;
    unsigned int s_mask_out_len = max_elems_per_block + 1;
    unsigned int s_merged_scan_mask_out_len = max_elems_per_block;
    unsigned int s_mask_out_sums_len = 4; // 4-way split
    unsigned int s_scan_mask_out_sums_len = 4;
    unsigned int shmem_sz = (s_data_len 
                            + s_mask_out_len
                            + s_merged_scan_mask_out_len
                            + s_mask_out_sums_len
                            + s_scan_mask_out_sums_len)
                            * sizeof(unsigned int);


    // for every 2 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    for (unsigned int shift_width = 0; shift_width <= 30; shift_width += 2)
    {
        gpu_radix_sort_local<<<grid_sz, block_sz, shmem_sz>>>(d_out, 
                                                                d_prefix_sums, 
                                                                d_block_sums, 
                                                                shift_width, 
                                                                d_in, 
                                                                d_in_len, 
                                                                max_elems_per_block);

        
        /*for(int ii=0; ii < d_in_len; ii++){
            std::cout << h_new1[ii] << "  ";
        }
        std::cout << std::endl;*/
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_block_sums, d_scan_block_sums, d_block_sums_len);

        // scan global block sum array
        //prefixsumScan(d_scan_block_sums, d_block_sums, d_block_sums_len);
        unsigned int* h_new = new unsigned int[d_block_sums_len];
        cudaMemcpy(h_new, d_scan_block_sums, sizeof(unsigned int) * d_block_sums_len, cudaMemcpyDeviceToHost);
        for(int ii=0; ii < d_in_len; ii++){
            std::cout << h_new[ii] << "  ";
        }
        std::cout << std::endl;
       
        // scatter/shuffle block-wise sorted array to final positions
        gpu_glbl_shuffle<<<grid_sz, block_sz>>>(d_in, 
                                                    d_out, 
                                                    d_scan_block_sums, 
                                                    d_prefix_sums, 
                                                    shift_width, 
                                                    d_in_len, 
                                                    max_elems_per_block);
        unsigned int* h_new1 = new unsigned int[d_in_len];
        cudaMemcpy(h_new1, d_out, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToHost);
      
    }
    cudaMemcpy(d_out, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToDevice);

    cudaFree(d_scan_block_sums);
    cudaFree(d_block_sums);
    cudaFree(d_prefix_sums);
}