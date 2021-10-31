#define BLOCK_SIZE 128
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 
#define THREADS_PER_BLOCK 512
#define ELEMENTS_PER_BLOCK 1024



__global__ void prescan(unsigned int *g_odata, unsigned int *g_idata, int n) 
{
    #if __CUDA_ARCH__ >= 200
    printf("1");
    #endif
    extern __shared__ unsigned int temp[];
    int thid = threadIdx.x;
    int offset = 1; 
    int ai = thid; 
    int bi = thid + (n/2); 
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    int g_index = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    temp[g_index] = 0;
    __syncthreads();
    #if __CUDA_ARCH__ >= 200
    printf("2");
    #endif
    if(g_index < n){
        temp[ai + bankOffsetA] = g_idata[ai];
        if(bi < n){
            temp[bi + bankOffsetB] = g_idata[bi];   
        }
    }
    for (int d = n>>1; d > 0; d >>= 1){ 
        __syncthreads();
        if (thid < d)
        { 
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            temp[bi] += temp[ai];    
        }    
        offset *= 2; 
    } 
    if (thid==0) 
    {
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
    }
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        { 
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            unsigned int t = temp[ai];
            temp[ai] = temp[bi]; 
            temp[bi] += t;       
        } 
    }  
    __syncthreads(); 
    g_odata[ai] = temp[ai + bankOffsetA];
    g_odata[bi] = temp[bi + bankOffsetB];
    #if __CUDA_ARCH__ >= 200
    printf("3");
    #endif
    
} 

void prefixsumScan(unsigned int *d_out, unsigned int *d_in, int length) {
    unsigned int shared_mem = BLOCK_SIZE + (BLOCK_SIZE >> LOG_NUM_BANKS);
	unsigned int blocks = length / 128;
    if(length%128 != 0){
        blocks++;
    }
    prescan<<<blocks, 128, sizeof(unsigned int) * shared_mem>>>(d_out, d_in, length);
}