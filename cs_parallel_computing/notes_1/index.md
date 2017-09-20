---
layout: default
---

## Parallel computing: Intro

Paralle computing surges in a need to get more computing power. In the early years, more computing power meant more CPU-speed, which meant to increase the clock of the processors. It worked from some time, but due to limitations in the design constraints of more recent computers, like power, etc., and also due to physical limitations in transistors as they are getting smaller and smaller ( quantum tunneling and that stuff ). 

I first read about how we got to this point in which we have nice tools for parallel computing in GPUs. Let me summarize it in a few lines and dates.

*  Early supercomputers had to increase its performance, so they also use that approach of increasing the clock speed of its processors. However, they also went into the path of using many processors instead of just one. This meant an increase in supercomputers' computing power, and this was then kind of taken into consideration when trying to develop consumer CPUs. More or less in 2015, CPUs started to be created with two computing cores instead of one. As the years went by, CPUs started to be made with three-, four-, six- and eight-core central processor units. This is were our CPUs are now, "Multicore".

*  In contrast to CPUs' processing pipelin, GPUs are quite different, and relatively new compared to their serial counterpart.

*  80s, 90s -> graphically driven operating systems made the necessity to create graphics accelerators. So in the 90s the first 2D display accelerators began to be made and purchased for personal computers. 

*  80s -> Silicon Graphics popularized 3D graphics.
*  1992 -> Silicon Graphics released the OpenGL library to talk to their hardware.
*  mid 90s -> 3D games ( first person shooters ) increased the need for more computing power to deal with this 3D graphics.
*  mid 90s -> Also, companies like NVIDIA started making graphics affordable graphics accelerators for personal computers.

*  Early shaders -> The early GPUs made by NVIDIA ( GeForce 3 series ) supported DirectX 8.0, which meant that the hardware should contain programmable vertex and pixel shading, allowing the developer to control these stages of the graphics pipeline.

*  Early 2000s -> Early GPU computing meant to know how to use OpenGL or DirectX to talk to the hardware. It allowed control over some phases of the graphics pipeline, but you had to transform your computation into a kind of graphics problem. This trick was useful, but kind of cumbersome, as you had to transform your data into something "like" color or pixels, and then write your computation into glsl or hlsl shading languages :(. Also, there were some restrictions in the kind of computations you could make, as some couldn't be transformed into graphics problems.

*  2006 -> NVIDIA released GeForce 8800 GTX, which supported DirectX 10 and also CUDA ( was built using the CUDA architecture ). This CUDA architecture had a unified shader pipeline ( not just vertex and pixel shaders, but general ). All ALUs on the chip could run general-purpose computations and supported IEEE single-precision floating point operations. Also, execution units on the GPU could arbitrarily write and read from memory and had software-managed access to a cache-like memory called shared memory.

*  2006 -> After the release of the NVIDIA GeForce 8800 GTX, NVIDIA added some features on top of standard C and made a public compiler for this language, CUDA C. They also allowed to communicate to the device by means of special drivers. So no more OpenGl or DirectX was needed to just communicate to the device and use it.


It's nice that we have now these tools for parallel computing on GPUs. When I first used a GPU was in the context of computer graphics, and was introduced to shaders. Before I was introduced to CUDA and OpenCL, I thought that only by means of shaders we could do computation on the GPU. Luckily I was wrong xD. Still, I recommend you to try some hello-world programs in OpenGL that use shaders to get an idea of how computation was made in that context, and then compare it with current CUDA code. I'll try to make a comparison here as well, maybe put some code and compare how to do two similar stuff in both the graphics and non-graphics contexts.


[back](./../..)
