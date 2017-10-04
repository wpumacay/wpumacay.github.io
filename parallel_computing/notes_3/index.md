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

This part is actually out of main, but it's better to take a look at it now. This is the code that will run in the device. You can ~~see~~ it by the "__global__" keyword. This qualifier tells the compiler to generate device code instead of host code. Here, the compiler sees the bifurcation and does what I mentioned. The CUDA compiler is actually a wrapper on top of the host compiler ( g++, for example ), which transfers the work to the host compiler when needed ( host code ) and transfering it to the CUDA compiler when needed ( device code ).

Appart from the return type, which is void, and the fancy "__global__" keyword, everything is the same as in any c++ function. It has input parameters, which in this case are two integers _a_ and _b_, and a pointer ( to device memory ) _pRes_. In the device code, we just use the variables and pointers the same way we would do in host code. There is no funky method like "cudaUseMemory" or something like that, although there are some special functions that deal with special cases, like "atomicAdd", which we will study later.

```c++
    // After the computation, get back the result
    cudaMemcpy( h_res, d_res, sizeof( int ), cudaMemcpyDeviceToHost );
```

Ok, so far we have requested the device to execute our add kernel, but the result is still in GPU world. We have to bring back that data, as we can dereference it in host code. 
Here we use another CUDA function, the "cudaMemcpy" function. It acts as a regular memcpy in host code, but the key difference is that it can deal with not only device to device copies, but also with host to device and device to host copies. In our case we want the data back from device memory to host memory, so the direction of copy will be deviceToHost ( like the last parameter in our function call, cudaMemcpyDeviceToHost ).

The parameters that we pass are :

1.  destination pointer, which is to where we want to copy the memory to. In our case is the host pointer _h_res_
2.  source pointer, which is from where we are going to copy the memory. It's our device pointer _d_res_ in this case.
3.  amount of memory to copy, which is the size of an int in our case as we are just using a single integer.
4.  the direction of the copy, which is cudaMemcpyDeviceToHost in our case.

Once completed, we will have the result in the block of memory pointed by _h_res_.

```c++
    // Print our result
    std::cout << "2 + 7 = " << *h_res << std::endl;
```

Here we just print the result. Nothing fancy.


```c++
    // Free the memory on the host
    free( h_res );
    // Free the memory on the device ( GPU )
    cudaFree( d_res );
```

Finally, once the data has served its purpose, we have to free it. We free the host memory by the call to the usual "free" in host code. We also have to free the memory in the GPU, which is done by the CUDA function "cudaFree". Of course, instead of passing a host pointer, we pass a device pointer.

When the program terminates all the memory used by the GPU is actually released, but we should call cudaFree when the memory is not longer necessary. As an example, imagine that we are dealing with block of memory in the GPU that are related to matrices. Once a matrix is no longer used, we should free its memory, as it could accumulate. Bad things could happen if we run out of memory. Memory leaks are a common issue in host c/c++ code, and they are also a issue in device code.

To compile our program we just have to call the "nvcc" compiler, like this :

```
nvcc -o add.out add.cu
```

This will compile our _add.cu_ file and generate the executable _add.out_. Let's just run it by calling :

```
./add.out
```

like any other regular executable. Of course, we should get this output :

```
    2 + 7 = 9
```

If you get anything different from 9 it could mean you are not memcopying correctly to the correct pointer. It can also mean that you have no CUDA capable GPU xD, as I did in my laptop which doesn't. You can actually have the cuda toolkit installed and compile code, but if no GPU is present you wont gen any useful result.

This takes us to talk a little about the device. We have some control over it, but how do we know the resources and limitations of our GPU, or if we have any GPU at all.

We will discuss this in the next section, as this section's notes are getting quite large. See you there.