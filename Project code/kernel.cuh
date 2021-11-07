#define BLOCK_SIZE 256
#include <cub/cub.cuh>
#include <iostream>
//NVIDIA prefix sum scan

__global__ void block_radix_sort(unsigned int* d_out_sorted,
    unsigned int* prefix_sums,
    unsigned int* d_block_sums,
    unsigned int shift_width,
    unsigned int* d_in,
    unsigned int d_in_len)
{

    //shared input array for a block
    extern __shared__ unsigned int shared_memory[];
    unsigned int* s_data = shared_memory;
    //shared mask array
    unsigned int* mask = &s_data[BLOCK_SIZE];
    unsigned int mask_len = BLOCK_SIZE + 1;
    //shared array for scan of mask on all 16 different radix
    unsigned int* merged_scan_mask = &mask[mask_len];
    //shared array for the sum of merged scan mask
    unsigned int* mask_sums = &merged_scan_mask[BLOCK_SIZE];
    //shared array for the scan of the mask sum
    unsigned int* s_scan_mask_sums = &mask_sums[16];
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
    //Digit from input of the current thread
    unsigned int t_data = s_data[thIdx];
    //Extracting radix of input depending on where we are in the loop
    unsigned int radix = (t_data >> shift_width) & 15;
    //mask = &s_data[128];
    for (unsigned int i = 0; i < 16; ++i)
    {
        // Zero out mask
        mask[thIdx] = 0;
        //To initialize last element of the mask
        if (thIdx == 0)
            mask[mask_len - 1] = 0;
        __syncthreads();

        // build bit mask output
        bool val_equals_i = false;
        //Set mask values depending on radix value
        if (cpy_idx < d_in_len)
        {
            val_equals_i = radix == i;
            mask[thIdx] = val_equals_i;
        }
        __syncthreads();
        
        //Hillis & Steele Parallel Scan Algorithm
        //Scan the mask array 
        unsigned int sum = 0;
        unsigned int max_steps = (unsigned int) log2f(BLOCK_SIZE);
        for (unsigned int d = 0; d < max_steps; d++) {
            if (thIdx < 1 << d) {
                sum = mask[thIdx];
            }
            else {
                sum = mask[thIdx] + mask[thIdx - (1 << d)];
                
            }
            __syncthreads();
            mask[thIdx] = sum;
            __syncthreads();
        }
        //Turn inclusive to exclusive scan
        unsigned int cpy_val;
        cpy_val = mask[thIdx];
        __syncthreads();
        mask[thIdx + 1] = cpy_val;
        __syncthreads();

        if (thIdx == 0)
        {
            // Transform exclusive scan to inclusive scan
            mask[0] = 0;
            unsigned int total_sum = mask[mask_len - 1];
            mask_sums[i] = total_sum;
            d_block_sums[i * gridDim.x + blockIdx.x] = total_sum;
        }
        __syncthreads();
        if (val_equals_i && (cpy_idx < d_in_len))
        {
            merged_scan_mask[thIdx] = mask[thIdx];
        }
        __syncthreads();
    }  

    // Scan on the mask output
    if (thIdx == 0)
    {
        unsigned int mask_sum = 0;
        for (unsigned int i = 0; i < 16; ++i)
        {
            s_scan_mask_sums[i] = mask_sum;
            mask_sum += mask_sums[i];
        }
    }

    __syncthreads();

    if (cpy_idx < d_in_len)
    {
        // Calculate the new indices of the input elements for sorting
        unsigned int t_prefix_sum = merged_scan_mask[thIdx];
        unsigned int new_pos = t_prefix_sum + s_scan_mask_sums[radix];
        
        __syncthreads();

        // Shuffle the block's input elements to actually sort them
        // Do this step for greater global memory transfer coalescing
        //  in next step
        s_data[new_pos] = t_data;
        merged_scan_mask[new_pos] = t_prefix_sum;
        
        __syncthreads();

        // Copy block - wise prefix sum results to global memory
        // Copy block-wise sort results to global 
        prefix_sums[cpy_idx] = merged_scan_mask[thIdx];
        d_out_sorted[cpy_idx] = s_data[thIdx];
    }
}

__global__ void block_shuffle(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* scan_block_sums,
    unsigned int* prefix_sums,
    unsigned int shift_width,
    unsigned int d_in_len)
{
    // get d = digit
    // get n = blockIdx
    // get m = local prefix sum array value
    // calculate global position = P_d[n] + m
    // copy input element to final position in d_out

    unsigned int thid = threadIdx.x;
    unsigned int cpy_idx = BLOCK_SIZE * blockIdx.x + thid;

    if (cpy_idx < d_in_len)
    {
        unsigned int t_data = d_in[cpy_idx];
        unsigned int global_radix = ((t_data >> shift_width) & 15) * gridDim.x + blockIdx.x;
        unsigned int data_glbl_pos = scan_block_sums[global_radix]+ prefix_sums[cpy_idx];
        __syncthreads();
        d_out[data_glbl_pos] = t_data;
    }
}

void radix_sort(unsigned int* const d_out,
    unsigned int* const d_in,
    unsigned int d_in_len)
{
    unsigned int grid_sz = d_in_len / BLOCK_SIZE;
    // Take advantage of the fact that integer division drops the decimals
    if (d_in_len % BLOCK_SIZE != 0)
        grid_sz += 1;

    unsigned int* prefix_sums;
    unsigned int prefix_sums_len = d_in_len;
    cudaMalloc(&prefix_sums, sizeof(unsigned int) * prefix_sums_len);
    cudaMemset(prefix_sums, 0, sizeof(unsigned int) * prefix_sums_len);

    unsigned int* d_block_sums;
    unsigned int d_block_sums_len = 16 * grid_sz; // 16-way split
    cudaMalloc(&d_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(d_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    unsigned int* scan_block_sums;
    cudaMalloc(&scan_block_sums, sizeof(unsigned int) * d_block_sums_len);
    cudaMemset(scan_block_sums, 0, sizeof(unsigned int) * d_block_sums_len);

    // shared memory consists of 3 arrays the size of the block-wise input
    //  and 2 arrays the size of n in the current n-way split (16)
    unsigned int s_data_len = BLOCK_SIZE;
    unsigned int mask_len = BLOCK_SIZE + 1;
    unsigned int merged_scan_mask_len = BLOCK_SIZE;
    unsigned int mask_sums_len = 16; // 16-way split
    unsigned int s_scan_mask_sums_len = 16;
    unsigned int shmem_sz = (s_data_len 
                            + mask_len
                            + merged_scan_mask_len
                            + mask_sums_len
                            + s_scan_mask_sums_len)
                            * sizeof(unsigned int);


    // for every 4 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    for (unsigned int shift_width = 0; shift_width <= 30; shift_width += 4)
    {
        block_radix_sort<<<grid_sz, BLOCK_SIZE, shmem_sz>>>(d_out, 
                                                                prefix_sums, 
                                                                d_block_sums, 
                                                                shift_width, 
                                                                d_in, 
                                                                d_in_len);


        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_block_sums, scan_block_sums, d_block_sums_len);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_block_sums, scan_block_sums, d_block_sums_len);

        // scan global block sum array
        //prefixsumScan(scan_block_sums, d_block_sums, d_block_sums_len);
        unsigned int* h_new = new unsigned int[d_block_sums_len];
        cudaMemcpy(h_new, scan_block_sums, sizeof(unsigned int) * d_block_sums_len, cudaMemcpyDeviceToHost);
       
        // scatter/shuffle block-wise sorted array to final positions
        block_shuffle<<<grid_sz, BLOCK_SIZE>>>(d_in, 
                                                    d_out, 
                                                    scan_block_sums, 
                                                    prefix_sums, 
                                                    shift_width, 
                                                    d_in_len);
        unsigned int* h_new1 = new unsigned int[d_in_len];
        cudaMemcpy(h_new1, d_out, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToHost);
      
    }
    cudaMemcpy(d_out, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToDevice);

    cudaFree(scan_block_sums);
    cudaFree(d_block_sums);
    cudaFree(prefix_sums);
}