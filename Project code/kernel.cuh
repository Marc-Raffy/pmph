#define BLOCK_SIZE 256
#include <cub/cub.cuh>
#include <iostream>

__global__ void radix_sort_block(unsigned int* d_out_sorted,
    unsigned int* block_prefix_sums,
    unsigned int* total_sum_block,
    unsigned int input_shift_width,
    unsigned int* d_in,
    unsigned int d_in_len,
    unsigned int block_size=BLOCK_SIZE)
{
    //shared input array for a block
    extern __shared__ unsigned int shared_memory[];
    unsigned int* s_data = shared_memory;
    //shared mask array
    unsigned int* mask = &s_data[block_size];
    unsigned int mask_len = block_size + 1;
    //shared array for scan of mask on all different radix
    unsigned int* s_merged_scan_mask = &mask[mask_len];
    //shared array for the sum of merged scan mask
    unsigned int* mask_sums = &s_merged_scan_mask[block_size];
    //histogram of radixes
    unsigned int* histogram = &mask_sums[16];
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
    unsigned int radix = (t_data >> input_shift_width) & 15;
    //mask = &s_data[128];
    for (unsigned int i = 0; i < 16; i++)
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
        unsigned int max_steps = (unsigned int) log2f(block_size);
        for (unsigned int d = 0; d < max_steps; d++) {
            if (thIdx < 1 << d) 
                sum = mask[thIdx];
            else 
                sum = mask[thIdx] + mask[thIdx - (1 << d)];
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
            //Inclusive to exclusive scan
            mask[0] = 0;
            unsigned int total_sum = mask[mask_len - 1];
            mask_sums[i] = total_sum;
            total_sum_block[i * gridDim.x + blockIdx.x] = total_sum;
        }
        __syncthreads();
        if (val_equals_i && (cpy_idx < d_in_len))
            s_merged_scan_mask[thIdx] = mask[thIdx];
        __syncthreads();
    }  

    // Exclusive naive scan on histogram
    if (thIdx == 0)
    {
        unsigned int temp_sum = 0;
        for (unsigned int i = 0; i < 16;i++)
        {
            histogram[i] = temp_sum;
            temp_sum += mask_sums[i];
        }
    }

    __syncthreads();

    if (cpy_idx < d_in_len)
    {
        //Get new indices
        unsigned int new_ind = s_merged_scan_mask[thIdx] + histogram[radix];
        __syncthreads();
        //Sort data in shared array for both input data and mask (merged and scanned version)
        s_data[new_ind] = t_data;
        s_merged_scan_mask[new_ind] = s_merged_scan_mask[thIdx];
        __syncthreads();
        //Write to global memory
        d_out_sorted[cpy_idx] = s_data[thIdx];
        block_prefix_sums[cpy_idx] = s_merged_scan_mask[thIdx];
        
    }
}

__global__ void block_shuffle(unsigned int* d_out,
    unsigned int* d_in,
    unsigned int* d_scan_block_sums,
    unsigned int* prefix_sums,
    unsigned int input_shift_width,
    unsigned int d_in_len,
    unsigned int block_size=BLOCK_SIZE)
{
    unsigned int thIdx = threadIdx.x;
    unsigned int cpy_idx = block_size * blockIdx.x + thIdx;

    if (cpy_idx < d_in_len)
    {
        unsigned int t_data = d_in[cpy_idx];
        unsigned int t_2bit_extract = (t_data >> input_shift_width) & 15;
        unsigned int t_prefix_sum = prefix_sums[cpy_idx];
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
    unsigned int grid_size = d_in_len / BLOCK_SIZE;

    unsigned int* prefix_sums;
    unsigned int prefix_sums_len = d_in_len;
    cudaMalloc(&prefix_sums, sizeof(unsigned int) * prefix_sums_len);
    cudaMemset(prefix_sums, 0, sizeof(unsigned int) * prefix_sums_len);

    unsigned int* block_sums;
    unsigned int block_sums_len = 16 * grid_size; // 4-way split
    cudaMalloc(&block_sums, sizeof(unsigned int) * block_sums_len);
    cudaMemset(block_sums, 0, sizeof(unsigned int) * block_sums_len);

    unsigned int* d_scan_block_sums;
    cudaMalloc(&d_scan_block_sums, sizeof(unsigned int) * block_sums_len);
    cudaMemset(d_scan_block_sums, 0, sizeof(unsigned int) * block_sums_len);

    // shared memory consists of 3 arrays the size of the block-wise input
    //  and 2 arrays the size of n in the current n-way split (4)
    unsigned int s_data_len = BLOCK_SIZE;
    unsigned int mask_len = BLOCK_SIZE + 1;
    unsigned int s_merged_scan_mask_len = BLOCK_SIZE;
    unsigned int mask_sums_len = 16; // 4-way split
    unsigned int histogram_len = 16;
    unsigned int shmem_sz = (s_data_len 
                            + mask_len
                            + s_merged_scan_mask_len
                            + mask_sums_len
                            + histogram_len)
                            * sizeof(unsigned int);


    // for every 2 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    for (unsigned int shift_width = 0; shift_width <= 28; shift_width += 4)
    {
        radix_sort_block<<<grid_size, BLOCK_SIZE, shmem_sz>>>(d_out, 
                                                                prefix_sums, 
                                                                block_sums, 
                                                                shift_width, 
                                                                d_in, 
                                                                d_in_len);

        //Perform a global exclusive scan on block sum                                 
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, block_sums, d_scan_block_sums, block_sums_len);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, block_sums, d_scan_block_sums, block_sums_len);

        unsigned int* h_new = new unsigned int[block_sums_len];
        cudaMemcpy(h_new, d_scan_block_sums, sizeof(unsigned int) * block_sums_len, cudaMemcpyDeviceToHost);
       
        // scatter/shuffle block-wise sorted array to final positions
        block_shuffle<<<grid_size, BLOCK_SIZE>>>(d_in, 
                                                    d_out, 
                                                    d_scan_block_sums, 
                                                    prefix_sums, 
                                                    shift_width, 
                                                    d_in_len);
        unsigned int* h_new1 = new unsigned int[d_in_len];
        cudaMemcpy(h_new1, d_out, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToHost);
      
    }
    cudaMemcpy(d_out, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToDevice);
}