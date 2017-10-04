---
layout: default
---

## Device properties

TODO


```c++
// LCommon.h
// Declarations of some helper functions
#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>


namespace common
{

    void printDeviceProps( int deviceId = -1 );

    void checkError( cudaError_t pErrorCode );

}



```

TODO

```c++

#include "LCommon.h"

using namespace std;

namespace common
{


    void printDeviceProps( int deviceId )
    {
        if ( deviceId == -1 )
        {
            int _devCount;
            // Check all devices
            checkError( cudaGetDeviceCount( &_devCount ) );

            for ( int q = 0; q < _devCount; q++ )
            {
                printDeviceProps( q );
            }
        }
        else
        {
            cudaDeviceProp _props;

            checkError( cudaGetDeviceProperties( &_props, deviceId ) );

            cout << " --- General information for device " << deviceId << " ---" << endl;
            cout << "Name: " << _props.name;
            cout << "Compute capability: " << _props.major << "." << _props.minor << endl;
            cout << "Clock rate: " << _props.clockRate << endl;
            cout << "Device copy overlap: " << ( ( _props.deviceOverlap ) ? "Enabled" : "Disabled" ) << endl;
            cout << "Kernel execution timeout: " << ( ( _props.kernelExecTimeoutEnabled ) ? "Enabled" : "Disabled" ) << endl;

            cout << " --- Memory information for device " << deviceId << " ---" << endl;
            cout << "Total global memory: " << _props.totalGlobalMem << endl;
            cout << "Total constant memory: " << _props.totalConstMem << endl;
            cout << "Max mem pitch: " << _props.memPitch << endl;
            cout << "Texture Alignment: " << _props.textureAlignment << endl;

            cout << " --- Multiprocessor information for device " << deviceId << " ---" << endl;
            cout << "Num multiprocessors: " << _props.multiProcessorCount << endl;
            cout << "Shared memory per multiprocessor: " << _props.sharedMemPerBlock << endl;
            cout << "Registers per multiprocessor: " << _props.regsPerBlock << endl;
            cout << "Threads in warp: " << _props.warpSize << endl;

            cout << "Max threads per block: " << _props.maxThreadsPerBlock << endl;
            cout << "Max thread dimensions: " << "( " 
                    << _props.maxThreadsDim[0] << " - "
                    << _props.maxThreadsDim[1] << " - "
                    << _props.maxThreadsDim[2] << " ) " << endl;
            cout << "Max grid dimensions: " << "( " 
                    << _props.maxGridSize[0] << " - "
                    << _props.maxGridSize[1] << " - "
                    << _props.maxGridSize[2] << " ) " << endl;
            cout << " ------------------------------------------------------------ " << endl;
        }
    }


    void checkError( cudaError_t pErrorCode )
    {
        if ( pErrorCode != cudaSuccess )
        {
            cerr << "Error: " << __FILE__ << " - " << __LINE__ << endl;
            cerr << "Code: " << pErrorCode << " - reason: " << cudaGetErrorString( pErrorCode ) << endl;

            exit( 1 );
        }
    }


}

```

TODO

```c++
#include "helpers/LCommon.h"

#include <iostream>

using namespace std;


int main()
{
    cout << "checking properties ..." << endl;

    common::printDeviceProps();

    cout << "done" << endl;

    return 0;
}
```

TODO

```
nvcc -o test_device.out test_device.cu helpers/LCommon.cpp

```

TODO

```
./test_device.out

checking properties ...
 --- General information for device 0 ---
Name: GeForce GTX 750 TiCompute capability: 5.0
Clock rate: 1254500
Device copy overlap: Enabled
Kernel execution timeout: Disabled
 --- Memory information for device 0 ---
Total global memory: 2096431104
Total constant memory: 65536
Max mem pitch: 2147483647
Texture Alignment: 512
 --- Multiprocessor information for device 0 ---
Num multiprocessors: 5
Shared memory per multiprocessor: 49152
Registers per multiprocessor: 65536
Threads in warp: 32
Max threads per block: 1024
Max thread dimensions: ( 1024 - 1024 - 64 ) 
Max grid dimensions: ( 2147483647 - 65535 - 65535 ) 
 ------------------------------------------------------------ 
done

```

TODO

```
./test_device.out 

checking properties ...
Error: helpers/LCommon.cpp - 66
Code: 35 - reason: CUDA driver version is insufficient for CUDA runtime version
```