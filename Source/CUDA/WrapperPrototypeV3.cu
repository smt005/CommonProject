
#include  "WrapperPrototypeV3.h"
#include <thread>
#include <vector>
#include <iostream>

#if ENABLE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <string>

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU ///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
    
namespace {
    __global__
        void GetForceForGPU(int* count, int* offset, CUDA::Body* bodies, CUDA::Vector3* forces) {
        float _constGravity = 0.01f;
        int statIndex = 0;// threadIdx.x;// +blockIdx.x * blockDim.x;
        int endIndex = *count;// statIndex + *offset;
        if (endIndex > *count) {
            endIndex = *count;
        }

        int sizeData = *count;
        float gravityVecX;
        float gravityVecY;
        float gravityVecZ;
        float dist;
        float force;

        for (int index = statIndex; index < endIndex; ++index) {
            for (size_t otherIndex = 0; otherIndex < sizeData; ++otherIndex) {
                if (index == otherIndex) {
                    continue;
                }

                gravityVecX = bodies[otherIndex].pos.x - bodies[index].pos.x;
                gravityVecY = bodies[otherIndex].pos.y - bodies[index].pos.y;
                gravityVecZ = bodies[otherIndex].pos.z - bodies[index].pos.z;

                dist = sqrt(gravityVecX * gravityVecX + gravityVecY * gravityVecY + gravityVecZ * gravityVecZ);
                gravityVecX /= dist;
                gravityVecY /= dist;
                gravityVecZ /= dist;

                force = _constGravity * (bodies[index].mass * bodies[otherIndex].mass) / (dist * dist);
                gravityVecX *= force;
                gravityVecY *= force;
                gravityVecZ *= force;

                forces[index].x += gravityVecX;
                forces[index].y += gravityVecY;
                forces[index].z += gravityVecZ;
            }
        }
    }
    
    __global__
        void GetForceGPU(int* count, CUDA::Body* bodyes, CUDA::Vector3* forces, int* devImin, int* devImax) {
        int index = threadIdx.x;// +blockIdx.x * blockDim.x;
        if (index >= *count) {
            return;
        }

        if (index < *devImin) {
            *devImin = index;
        }
        if (index > *devImax) {
            *devImax = index;
        }

        double _constGravity = 0.01f;
        int sizeData = *count;
        float gravityVecX = 0;
        float gravityVecY = 0;
        float gravityVecZ = 0;
        float dist;
        float force;
        float mass = bodyes[index].mass;

        float posX = bodyes[index].pos.x;
        float posY = bodyes[index].pos.y;
        float posZ = bodyes[index].pos.z;

        for (int otherIndex = 0; otherIndex < sizeData; ++otherIndex) {
            if (index == otherIndex) {
                continue;
            }

            gravityVecX = bodyes[otherIndex].pos.x - posX;
            gravityVecY = bodyes[otherIndex].pos.y - posY;
            gravityVecZ = bodyes[otherIndex].pos.z - posZ;

            dist = sqrt(gravityVecX * gravityVecX + gravityVecY * gravityVecY + gravityVecZ * gravityVecZ);
            gravityVecX /= dist;
            gravityVecY /= dist;
            gravityVecZ /= dist;

            force = _constGravity * (mass * bodyes[otherIndex].mass) / (dist * dist);
            gravityVecX *= force;
            gravityVecY *= force;
            gravityVecZ *= force;

            forces[index].x += gravityVecX;
            forces[index].y += gravityVecY;
            forces[index].z += gravityVecZ;
        }
    }
}

int CUDA_PrototypeV3::tag = 0;
int CUDA_PrototypeV3::tagCurrent = -1;

void CUDA_PrototypeV3::GetForcesGPUStatic(std::vector<CUDA::Body>& bodies, std::vector<CUDA::Vector3>& forces) {
    int count = bodies.size();
    forces.resize(count, CUDA::Vector3());

    unsigned int counThread = count < CUDA::maxThreadsPerBlock ? count : CUDA::maxThreadsPerBlock;

    unsigned int countBlock = (count + counThread - 1) / counThread;
    countBlock = countBlock > CUDA::maxGridSize[1] ? CUDA::maxGridSize[1] : countBlock;

    int offset = count / (counThread * countBlock);
    if ((count % (counThread * countBlock)) > 0) {
        ++offset;
    }

    //...
    int* devCount;
    int* devOffset;
    CUDA::Body* devBodyes;
    CUDA::Vector3* devForces;

    cudaError_t error;

    cudaMalloc(&devCount,               sizeof(int));
    cudaMalloc(&devOffset,              sizeof(int));
    cudaMalloc(&devBodyes,      count * sizeof(CUDA::Body));
    cudaMalloc(&devForces,      count * sizeof(CUDA::Vector3));

    int iMin = 10000;
    int iMax = -10000;
    int* devImin;
    int* devImax;

    cudaMalloc(&devImin, sizeof(int));
    cudaMalloc(&devImax, sizeof(int));

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA_PrototypeV3 cudaMalloc error: %s\n", cudaGetErrorString(error));
        return;
    }

    cudaMemcpy(devCount,        &count,              sizeof(int),           cudaMemcpyHostToDevice);
    cudaMemcpy(devOffset,       &offset,             sizeof(int),           cudaMemcpyHostToDevice);
    cudaMemcpy(devBodyes, bodies.data(),     count * sizeof(CUDA::Body), cudaMemcpyHostToDevice);
    cudaMemcpy(devForces, bodies.data(),    count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);

    cudaMemcpy(devImin, &iMin, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devImax, &iMax, sizeof(int), cudaMemcpyHostToDevice);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA_PrototypeV3 cudaMemcpy error: %s\n", cudaGetErrorString(error));
        return;
    }

    if (tag == 0) {
        if (tag != tagCurrent) {
            tagCurrent = tag;
            printf("GPU: [%i] CUDA_PrototypeV3::GetForceForGPU\n", tagCurrent);
        }
        //GetForceForGPU <<<countBlock, counThread>>> (devCount, devOffset, devBodyes, devForces);
        GetForceForGPU << <1, 1 >> > (devCount, devOffset, devBodyes, devForces);

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA_PrototypeV3 GetForceForGPU error: %s\n", cudaGetErrorString(error));
            return;
        }
    }
    else if (tag == 1) {
        if (tag != tagCurrent) {
            tagCurrent = tag;
            printf("GPU: [%i] CUDA_PrototypeV3::GetForceGPU\n", tagCurrent);
        }
        //GetForceGPU <<<countBlock, counThread>>> (devCount, devBodyes, devForces);
        GetForceGPU << <1, count >> > (devCount, devBodyes, devForces, devImin, devImax);

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA_PrototypeV3 GetForceGPU error: %s\n", cudaGetErrorString(error));
            return;
        }
    }

    cudaMemcpy(forces.data(), devForces, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);

    cudaMemcpy(&iMin, devImin, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&iMax, devImax, sizeof(int), cudaMemcpyDeviceToHost);

    if (iMin < 0 || iMax >= count) {
        printf("CUDA_PrototypeV3 error count: [%i, %i]\n", iMin, iMax);
    }

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA_PrototypeV3 cudaMemcpy error: %s\n", cudaGetErrorString(error));
        return;
    }

    cudaFree(devCount);
    cudaFree(devOffset);
    cudaFree(devBodyes);
    cudaFree(devForces);

    cudaFree(devImin);
    cudaFree(devImax);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA_PrototypeV3 cudaFree error: %s\n", cudaGetErrorString(error));
        return;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU ///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
    void GetForceCPU(int* count, int* offset, CUDA::Body* bodies, CUDA::Vector3* forces, int threadId) {
        float _constGravity = 0.01f;
        int statIndex = 0;// threadIdx.x;// +blockIdx.x * blockDim.x;
        int endIndex = *count;// statIndex + *offset;
        if (endIndex > *count) {
            endIndex = *count;
        }

        int sizeData = *count;
        float gravityVecX;
        float gravityVecY;
        float gravityVecZ;
        float dist;
        float force;

        for (int index = statIndex; index < endIndex; ++index) {
            for (size_t otherIndex = 0; otherIndex < sizeData; ++otherIndex) {
                if (index == otherIndex) {
                    continue;
                }

                gravityVecX = bodies[otherIndex].pos.x - bodies[index].pos.x;
                gravityVecY = bodies[otherIndex].pos.y - bodies[index].pos.y;
                gravityVecZ = bodies[otherIndex].pos.z - bodies[index].pos.z;

                dist = sqrt(gravityVecX * gravityVecX + gravityVecY * gravityVecY + gravityVecZ * gravityVecZ);
                gravityVecX /= dist;
                gravityVecY /= dist;
                gravityVecZ /= dist;

                force = _constGravity * (bodies[index].mass * bodies[otherIndex].mass) / (dist * dist);
                gravityVecX *= force;
                gravityVecY *= force;
                gravityVecZ *= force;

                forces[index].x += gravityVecX;
                forces[index].y += gravityVecY;
                forces[index].z += gravityVecZ;
            }
        }
    }
}

void CUDA_PrototypeV3::GetForcesCPUStatic(std::vector<CUDA::Body>& bodies, std::vector<CUDA::Vector3>& forces) {
    int counThread = static_cast<double>(std::thread::hardware_concurrency());
    int count = bodies.size();
    forces.resize(count, CUDA::Vector3());

    /*if ((count * 2) > counThread) {
        int offst = count / counThread;
        if ((count % counThread) > 0) {
            ++offst;
        }

        std::vector<std::thread> threads;
        threads.reserve(counThread);

        for (int threadId = 0; threadId < counThread; ++threadId) {
            threads.emplace_back([&]() {
                GetForceCPU(&count, &offst, bodies.data(), forces.data(), threadId);
            });
        }

        for (std::thread& th : threads) {
            th.join();
        }
    }
    else*/
    {
        GetForceCPU(&count, &count, bodies.data(), forces.data(), 0);
    }
}

//...
#else
    void testCUDA(void) {}

    void CUDA::GetProperty() {}
    void CUDA::PrintInfo() {}
#endif
