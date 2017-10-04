---
layout: default
---

# [](#header-1) Summary

This section will try to summarize the notes covered in each section. It was actually an attempt to review some
concepts for an exam I had to take. Hope it's useful for you.

## CUDA Memory overview

So, there are some key ideas we have to remember about memory in CUDA, and those are:

1.   Global memory
2.   Shared memory
3.   Local memory
4.   Constant memory
5.   Texture memory
6.   Pinned memory ( page-locked )
7.   Zero-copy memory ( a variant of pinned memory )

Let's go over them one by one.


## Global memory

This is the big chunk of memory that you have available for computations. You 
usually deal with it when doing cudaMallocs and cudaMemcpys. This memory resides in DDR RAM 
that resides in the device, but not in the GPU-chip itself. There are huge amounts of this memory
compared with the other types ( in the order of GBs ). To find out how much, just query the
properties of your device. This is the base memory we use, and is usually the slower, by more or less
two orders of magnitude compared with its cached counterpart Shared memory, and local registers.

All cudaMemcpys basically store data into this block of memory, and as you recall, you define pointers in
host code to deal with the cuda calls to handle this memory.

```c++
    float* d_vec;
    cudaMalloc( ( void** )&d_vec, size );
    cudaMemcpy( d_vec, someHostData, size, cudaMemcpyHostToDevice );
```

It's usually when dealing with this memory that we have the issue of potentially dereferencing the pointers
created for the global memory allocated in GPU in host code. 
Also, Pinned and Zero-copy memory are actual blocks of global memory, with some extra functionality, and texture
memory resides actually in global memory but has some other features.

## Shared memory

This is basically the local memory that can be used by working threads in the same block. If you recall some of the architecture 
of a CUDA GPU, it's composed of lots of CUDA cores that are arranged in groups calles Streaming Multiprocessors ( SMs ). This are
collections of physical compute cores and basically the blocks have to be scheduled in such a way that they fit into an SM. It's in
this SM that we have a cache-like memory that we can use to share data between working threads in a same SM.

This memory is quite fast, faster that global memory by a factor of approx. 100 times, but sure it limited in size. If you query your
device you will see that you have just some dozens of KBs available for each SM.

To work with this kind of memory, we declare it inside device code, as opposed to global memory, which is declared and reserved in host code.

```c++
    __shared__ float sh_foo[100];
```

The amount of memory has to be know at compile time if declared in this way. It could be dynamic if we pass the amount of memory in the kernel call.

```c++
    fooKernel<<< grid, block, sizeOfSharedMemory >>>( args )
```

In this context we have to deal with synchronization calls, like __syncthreads(), which need to be called in order for the threads in a block to
synchronize its execution ( wait till all have reached that section ).

## Local memory

This is the memory used to store registers or local primitive variables, like floats, ints, etc. inside device code. It's faster that global memory
more or less in the same order of magnitude as shared memory.

## Constant memory

This is a special kind of global memory. When we declare some region as __constant__, we ask the compiler to treat this region as read-only. This
has some advantages, as the memory can be cached efficiently because wont change, and also less read requests within a same half-warp ( more on warps in some sections )
by using a single read and then a broadcast, reducing the memory transfers by a good order of magnitude.

To define a region as constant we declare like this

```c++
    __constant__ float foo[SIZE];
```

To actually fill this region, we use _cudaMemcpyToSymbol_ instead of a normal _cudaMemcpy_.

## Texture memory

This is memory similar to constant memory, in the sense that is read-only, but it actually resides on global memory. The way it improves is that is
accessed by a special read-only data line and it can also be cached. To use or not to use it depends of the problem form. Problems that can take advantage of
texture memory are the ones that do computation in regions close to each other in a 2D space, to name an example.

To create and use it we have to first define a texture object

```c++
    texture<float, 2> myTexture;
```

Then bind it to a region already allocated with cudaMalloc

```c++
    cudaBindTexture( ... );
```

And then we can use it in our kernel code by using tex2Dfetch or tex1Dfetch, according to the dimension of our texture.

## Pinned memory

This is a special kind of global memory in the sense that it won't be paged and sent to the swap region ( virtual memory on disk ) of our computer. Instead, it
is force to remain in RAM always.


## Streams

