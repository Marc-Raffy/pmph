#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#include "transpose-kernels.cu.h"
#include "transpose-host.cu.h"

#define HEIGHT_A 1024*8   //12835//2048//2048
#define  WIDTH_A 1024*8  //15953 //1024//2048
#define TILE     32
#define RUNS_GPU 100

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}


/**
 * Measure a more-realistic optimal bandwidth by a simple, memcpy-like kernel
 */ 
int bandwidthMemcpy( const uint32_t B     // desired CUDA block size ( <= 1024, multiple of 32)
                   , const size_t   N     // length of the input array
                   , float* d_in          // device input  of length N
                   , float* d_out         // device result of length N
) {
    // dry run to exercise the d_out allocation!
    const uint32_t num_blocks = (N + B - 1) / B;
    naiveMemcpy<<< num_blocks, B >>>(d_out, d_in, N);
    
    double gigaBytesPerSec;
    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    { // timing the GPU implementations
        gettimeofday(&t_start, NULL); 

        for(int i=0; i<RUNS_GPU; i++) {
            naiveMemcpy<<< num_blocks, B >>>(d_out, d_in, N);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU;
        gigaBytesPerSec = 2 * N * sizeof(int) * 1.0e-3f / elapsed;
        printf("Naive Memcpy GPU Kernel runs in: %lu microsecs, GB/sec: %.2f\n\n\n"
              , elapsed, gigaBytesPerSec);
    }
 
    gpuAssert( cudaPeekAtLastError() );
    return 0;
}

void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}


template<class T>
void matTranspose(T* A, T* trA, int rowsA, int colsA) {
  for(int i = 0; i < rowsA; i++) {
    for(int j = 0; j < colsA; j++) {
      trA[j*rowsA + i] = A[i*colsA + j];
    }
  }
}

template<class T>
bool validateTranspose(float* A,float* trA, unsigned int rowsA, unsigned int colsA){
  bool valid = true;
  for(unsigned int i = 0; i < rowsA; i++) {
    for(unsigned int j = 0; j < colsA; j++) {
      if(trA[j*rowsA + i] != A[i*colsA + j]) {
        printf("row: %d, col: %d, A: %.4f, trA: %.4f\n", 
                i, j, A[i*colsA + j], trA[j*rowsA + i] );
        valid = false;
        break;
      }
    }
    if(!valid) break;
  }
  if (valid) printf("GPU TRANSPOSITION   VALID!\n");
  else       printf("GPU TRANSPOSITION INVALID!\n");
  return valid;
}


bool validateProgram(float* A, float* B, unsigned int N){
  bool valid = true;
  for(unsigned int i = 0; i < N; i++) {
    unsigned long long ii = i*64;
    double accum = 0.0;
    for(int j = 0; j < 64; j++) {
        float tmpA  = A[ii + j];
        accum = sqrt(accum) + tmpA*tmpA;
        if(fabs(B[ii+j] - accum) > 0.00001) {
            printf("Row %d column: %d, seq: %f, par: %f\n", i, j, accum, B[ii+j]);
            valid = false; break; 
        }
    }
    if(!valid) break;
  }
  if (valid) printf("GPU PROGRAM   VALID!\n");
  else       printf("GPU PROGRAM INVALID!\n");
  return valid;
}


int weekly3Task3( int height
                , float* h_A
                , float* h_B
                , float* d_A
                , float* d_B
) {
    const uint32_t width = 64; // each row has 64 float elements
    const size_t mem_size = height * width * sizeof(float);
    const unsigned int REPEAT = RUNS_GPU;
    double gigaBytesPerSec;

    { // compute original program
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        unsigned int block      = 256;
        unsigned int num_thds   = height;
        unsigned int num_blocks = (height + block - 1) / block;

        for (int kkk = 0; kkk < REPEAT; kkk++) {
            origProg<<<num_blocks, block>>>(d_A, d_B, num_thds);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / REPEAT; 
        gigaBytesPerSec = 2 * mem_size * 1.0e-3f / elapsed;
        printf( "Original Program runs on GPU in: %lu microsecs, GB/sec: %f\n"
              , elapsed, gigaBytesPerSec);

        // copy result from device to host
        cudaMemcpy(h_B, d_B, mem_size, cudaMemcpyDeviceToHost);
        gpuAssert( cudaPeekAtLastError() );
        cudaMemset(d_B, 0, mem_size);
        validateProgram(h_A, h_B, num_thds);
    }

    { // Compute transformed program in which all read and write 
      // accesses are coalesced. This is obtained by transposing
      // input array A and result array B.

        // We allocate device buffers for d_Atr---i.e., the transposed of A,
        // and for d_Btr---i.e., the transpose of B
        float* d_Atr;   cudaMalloc((void**) &d_Atr, mem_size);
        float* d_Btr;   cudaMalloc((void**) &d_Btr, mem_size);
 
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        unsigned int num_thds   = height;
        unsigned int block      = 256;
        unsigned int num_blocks = (num_thds + block - 1) / block;

        // Task3.a) ToDo: 
        //       - fill in the implementation CPU orchestration code below
        //       - and the corresponding CUDA kernel!
        for (int kkk = 0; kkk < REPEAT; kkk++) {
            // 3.a.1  you probably need to transpose d_A here by
            //        using function "transposeTiled<float, TILE>"
            //        i.e., source array is d_A, result array is d_Atr
            transposeTiled<float, TILE>( d_A, d_Atr, height, width );
            // 3.a.2  you probably need to implement the "transfProg"
            //        kernel in file transpose-kernel.cu.h which takes 
            //        input from d_Atr, and writes the result in d_Btr,
            transfProg<<< num_blocks, block >>>(d_Atr, d_Btr, num_thds);
            // 3.a.3  you probably need to transpose-back the result here
            //        i.e., source array is d_Btr, and transposed result
            //        is in d_B.
            transposeTiled<float, TILE>( d_Btr, d_B, width, height );
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / REPEAT; 
        gigaBytesPerSec = 2 * mem_size * 1.0e-3f / elapsed;
        printf( "Coalesced Program with manifested transposition runs on GPU in: %lu microsecs, GB/sec: %f\n"
              , elapsed, gigaBytesPerSec);

        // copy result from device to host
        cudaMemcpy(h_B, d_B, mem_size, cudaMemcpyDeviceToHost);
        gpuAssert( cudaPeekAtLastError() );
        cudaMemset(d_B, 0, mem_size);

        validateProgram(h_A, h_B, num_thds);

        // deallocate the transposed buffers here
        cudaFree(d_Atr);
        cudaFree(d_Btr);
   }

    { // Optimized program---i.e., exhibiting only coalesced 
      // accesses---obtained by using the shared memory as
      // a staging buffer, i.e., read from global-to-shared
      // memory (in coalesced way) and then each thread reads
      // from shared memory in non-coalesced way. Note that
      // this version should be the fastest, as it does not
      // require to perform (manifest) the transpositions
      // (in global memory).
      // 
      // Task 3.b) implement function "glb2shmem" in file
      //           "transpose-kernels.cu.h"
        cudaMemset(d_B, 0, mem_size);

        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        unsigned int num_thds= height;
        unsigned int block   = 256;
        unsigned int grid    = (num_thds + block - 1) / block;
        const int CHUNK = 16;

        if((block % CHUNK) != 0) {
            printf("Broken Assumption: block size not a multiple of chunk size, EXITING!\n");
            exit(1);
        }

        for (int kkk = 0; kkk < REPEAT; kkk++) {
            optimProg<CHUNK><<<grid, block, CHUNK*block*sizeof(float)>>>(d_A, d_B, num_thds);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / REPEAT; 
        gigaBytesPerSec = 2 * mem_size * 1.0e-3f / elapsed;
        printf("Optimized Program runs on GPU in: %lu microsecs, GB/sec: %f\n", elapsed, gigaBytesPerSec);

        gpuAssert( cudaPeekAtLastError() );

        // copy result from device to host
        cudaMemcpy(h_B, d_B, mem_size, cudaMemcpyDeviceToHost);
        cudaMemset(d_B, 0, mem_size);
        validateProgram(h_A, h_B, num_thds);
    }
    return 0;
}


int main() {
    // set seed for rand()
    srand(2006);
 
    // 1. allocate host memory for the two matrices
    size_t size_A = WIDTH_A * HEIGHT_A;
    size_t mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    float* h_B = (float*) malloc(mem_size_A);
 
    // 2. initialize host memory
    randomInit(h_A, size_A);
    
    // 3. allocate device memory
    float* d_A;
    float* d_B;
    cudaMalloc((void**) &d_A, mem_size_A);
    cudaMalloc((void**) &d_B, mem_size_A);
 
    // 4. copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    bandwidthMemcpy( 256, size_A, d_A, d_B );
    gpuAssert( cudaPeekAtLastError() );

    { // test transpose
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        for (int kkk = 0; kkk < RUNS_GPU; kkk++) {
            //transposeNaive<float, TILE>( d_A, d_B, HEIGHT_A, WIDTH_A );
            transposeTiled<float, TILE>( d_A, d_B, HEIGHT_A, WIDTH_A );
        }
        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / RUNS_GPU; 
        double gigaBytesPerSec = 2.0 * mem_size_A * 1.0e-3f / elapsed;
        printf("Transpose on GPU runs in: %lu microsecs, GB/sec: %f\n", elapsed, gigaBytesPerSec);

        // copy result from device to host
        cudaMemcpy(h_B, d_B, mem_size_A, cudaMemcpyDeviceToHost);
  
        // validate
        validateTranspose<float>( h_A, h_B, HEIGHT_A, WIDTH_A );
    }
    gpuAssert( cudaPeekAtLastError() );

    weekly3Task3( WIDTH_A * HEIGHT_A / 64, h_A, h_B, d_A, d_B );

   // clean up memory
   free(h_A);
   free(h_B);
   cudaFree(d_A);
   cudaFree(d_B);
}

