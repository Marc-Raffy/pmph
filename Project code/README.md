# Fast Radix Sort
Code is separated rated in futhark and CUDA codes.
Futhark code is based on calling the DIKU radix sort implementation.
CUDA code is calling both our CUDA radix sort implementation, CUB library and sort algorithm from STD.

# Futhark code
For futhark code it depends on the futhark radix sort implementation on the following website
[Sorts DIKU algorithms](https://github.com/diku-dk/sorts/actions).
This package contains various sorting algorithms implemented in Futhark.

## Installation
Maybe need steps:
```
$ futhark pkg add github.com/diku-dk/sorts
$ futhark pkg sync
```

## Usage
For the Futhark we need to run in manually.
First compile and create a directory for storing the runtime measurements.
And then we have different runs on specified input sizes (2^17-2^30).
```
$ futhark opencl radix_sort_futhark.fut
$ mkdir runtimes
$ futhark dataset --i32-bounds=0:131072 -s 73 -b -g [131072]i32 | ./radix_sort_futhark -t runtimes/runtime17.txt -r 10
$ futhark dataset --i32-bounds=0:262144 -s 73 -b -g [262144]i32 | ./radix_sort_futhark -t runtimes/runtime18.txt -r 10
$ futhark dataset --i32-bounds=0:524288 -s 73 -b -g [524288]i32 | ./radix_sort_futhark -t runtimes/runtime19.txt -r 10
$ futhark dataset --i32-bounds=0:1048576 -s 73 -b -g [1048576]i32 | ./radix_sort_futhark -t runtimes/runtime20.txt -r 10
$ futhark dataset --i32-bounds=0:2097152 -s 73 -b -g [2097152]i32 | ./radix_sort_futhark -t runtimes/runtime21.txt -r 10
$ futhark dataset --i32-bounds=0:4194304 -s 73 -b -g [4194304]i32 | ./radix_sort_futhark -t runtimes/runtime22.txt -r 10
$ futhark dataset --i32-bounds=0:8388608 -s 73 -b -g [8388608]i32 | ./radix_sort_futhark -t runtimes/runtime23.txt -r 10
$ futhark dataset --i32-bounds=0:16777216 -s 73 -b -g [16777216]i32 | ./radix_sort_futhark -t runtimes/runtime24.txt -r 10
$ futhark dataset --i32-bounds=0:33554432 -s 73 -g [33554432]i32 | ./radix_sort_futhark -t runtimes/runtime25.txt -r 10
$ futhark dataset --i32-bounds=0:67108864 -s 73 -g [67108864]i32 | ./radix_sort_futhark -t runtimes/runtime26.txt -r 10
$ futhark dataset --i32-bounds=0:134217728 -s 73 -b -g [134217728]i32 | ./radix_sort_futhark -t runtimes/runtime27.txt -r 10
$ futhark dataset --i32-bounds=0:268435456 -s 73 -b -g [268435456]i32 | ./radix_sort_futhark -t runtimes/runtime28.txt -r 10
$ futhark dataset --i32-bounds=0:536870912 -s 73 -b -g [536870912]i32 | ./radix_sort_futhark -t runtimes/runtime29.txt -r 10
$ futhark dataset --i32-bounds=0:1073741824 -s 73 -b -g [1073741824]i32 | ./radix_sort_futhark -t runtimes/runtime30.txt -r 10
```
# CUDA code
The CUDA code is run in the `main.cu` which has the test for the runtimes.
The output shows the runtimes for input sizes (2^17-2^30) and shows the times for the respectively:
    
- CPU sequential sort from STD library
- Our CUDA radix sort implementation
- Radix sort from CUB library

## Installation
There shouldnt be anything additional to install

## Usage
To run CUDA code just do:
```
$ make
```