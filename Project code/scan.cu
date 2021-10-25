#define BLOCK_SIZE 128
#define NUM_BANKS 16 
#define LOG_NUM_BANKS 4 
#define CONFLICT_FREE_OFFSET(n) \
((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 
int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

__global__ void prescan(unsigned int *g_odata, unsigned int *g_idata, int n) { 
    extern __shared__ float temp[];
    int thid = threadIdx.x;
    int offset = 1; 
       int ai = thid; 
       int bi = thid + (n/2); 
       int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
       int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
       temp[ai + bankOffsetA] = g_idata[ai];
       temp[bi + bankOffsetB] = g_idata[bi]; 
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
            float t = temp[ai];
            temp[ai] = temp[bi]; 
            temp[bi] += t;       
        } 
    }  
    __syncthreads(); 
    g_odata[ai] = temp[ai + bankOffsetA];
    g_odata[bi] = temp[bi + bankOffsetB]; 
} 

void scanLargeEvenDeviceArray(unsigned int *d_out, unsigned int *d_in, int length) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

		prescan<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK);
}