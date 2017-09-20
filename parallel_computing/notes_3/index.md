---
layout: default
---

## Hello World: Intro to CUDA programming

Well, to get started let's do a basic hello world program in CUDA. Not an actual "print 'helloWorld'", but an add two numbers example. Should be basic enough and give you the basic of how to write a CUDA program.

```c++
// add.cu
// Add two number example

#include <iostream>

__global__ void add( int a, int b, int *pRes )
{
    *pRes = a + b;
}

int main()
{
    // Create some pointers for GPU memory and cpu memory
    int *d_res;
    int *h_res;

    // Allocate some memory in the GPU, just one int for our result 
    cudaMalloc( ( void** )&d_res, sizeof( int ) );

    // Also, allocate memory for the result, to get it back from our GPU
    h_res = ( int* ) malloc( sizeof( int ) );

    // Call our add "function" ( function, but fancy named "kernel" )
    add<<< 1, 1 >>>( 2, 7, d_res );

    // After the computation, get back the result
    cudaMemcpy( h_res, d_res, sizeof( int ), cudaMemcpyDeviceToHost );

    // Print our result
    std::cout << "2 + 7 = " << *h_res << std::endl;

    // Free the memory on the host
    free( h_res );
    // Free the memory on the device ( GPU )
    cudaFree( d_res );
}

```

Let's analyze the parts of the code. Let's start by the main method.


```c++
    int *d_res;
    int *h_res;
```

In this part we just create two pointers. As we will see later, they will point to different kinds of memory :D. The _h_res_ pointer will point to memory in our host computer ( that's why the 'h_' in the variable name ), and the other pointer, _d_res_, will point to memory in our device ( our GPU, that's why the 'd_' ), which is a big fat chunk of global memory in the GPU, around 2GB, depending of your GPU ( mine, a GTX 750 Ti has just 2GB :'( );

It's a good practice to keep track of the pointers we use with this 'd_' and 'h_' notation, as will make it clear to which kind of data they are pointing to. The most common error you could run into is trying to dereference the memory pointed by a device pointer, which will make our program crash spectacularly xD ( well, the most annoying bugs to me are the ones that freeze your computer, but this is also really annoying, like forgetting a ";" ).

```c++
    // Allocate some memory in the GPU, just one int for our result 
    cudaMalloc( ( void** )&d_res, sizeof( int ) );
```

This part is a call to the CUDA API, specifically the 'cudaMalloc' function, which is similar to the old good malloc in host code. Like malloc, is in charge or reserving memory, but in this case in the device ( GPU ) and "returns" ( it writes the direction to ) a pointer to that reserved block of memory.

```c++
    // Also, allocate memory for the result, to get it back from our GPU
    h_res = ( int* ) malloc( sizeof( int ) );
```

This is the old good malloc, which just reserves memory on the host, and returns a pointer to the block reserved.

```c++
    // Call our add "function" ( function, but fancy named "kernel" )
    add<<< 1, 1 >>>( 2, 7, d_res );
```

This is the fun part. Here we request the runtime to execute our "kernel", called "add", in the GPU. You see there are some extra stuff that make this call different to a simple call to a c++ host function, which are the "<<<", ">>>" brackets. This is a special qualifier that tells the runtime to execute our kernel.

You should be wondering what are those parameters inside "<<< >>>". Well, if you recall a GPU has MANY cores, which mean we can execute our device code into MANY cores. The parameters passed to that qualifier are the way of controlling how many of those core are going to run our device code and also how they should distribute the work. We will see more on how the work is distributed in the notes that deal with the GPU Architecture, but will start manipulating these parameters and explain the basics of how to distribute work in the GPU in the next notes.

In our case, just 1 "thread" is requested. The GPU will use its magic to know which core should run this thread, so we don't have to worry about that. It's kind of like a scheduler in an Operating System.


```c++
__global__ void add( int a, int b, int *pRes )
{
    *pRes = a + b;
}
```

