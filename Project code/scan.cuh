#define BLOCK_SIZE 128
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#define THREADS_PER_BLOCK 512
#define ELEMENTS_PER_BLOCK 256

__global__ void prescan(unsigned int *g_odata, unsigned int *g_idata, int n)
{
    extern __shared__ float temp[]; // allocated on invocation

    int thid = threadIdx.x;
    int offset = 1;

    temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
    temp[2 * thid + 1] = g_idata[2 * thid + 1];

    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();

        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0)
    {
        temp[n - 1] = 0;
    } // clear the last element

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;

        __syncthreads();

        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();

    g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
    g_odata[2 * thid + 1] = temp[2 * thid + 1];
}

void prefixsumScan(unsigned int *d_out, unsigned int *d_in, int length)
{
    unsigned int shared_mem = BLOCK_SIZE + (BLOCK_SIZE >> LOG_NUM_BANKS);
    unsigned int blocks = length / 128;
    if (length % 128 != 0)
    {
        blocks++;
    }
    prescan<<<blocks, 128, sizeof(float) * shared_mem>>>(d_out, d_in, ELEMENTS_PER_BLOCK);
}