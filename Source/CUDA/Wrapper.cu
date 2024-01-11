
#include "Wrapper.h"
#include <thread>
#include <vector>

std::string CUDA::nameGPU;
int         CUDA::deviceCount = -1;
int         CUDA::warpSize = 0;
int         CUDA::maxThreadsPerBlock = 0;
int         CUDA::maxThreadsDim[3];
int         CUDA::maxGridSize[3];
int         CUDA::maxThreadsPerMultiProcessor = 0;
int         CUDA::maxBlocksPerMultiProcessor = 0;

#if ENABLE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

void CUDA::GetProperty() {
    if (deviceCount == -1) {
        cudaGetDeviceCount(&deviceCount);

        if (deviceCount == 0) {
            return;
        }

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        nameGPU = deviceProp.name;
        warpSize = deviceProp.warpSize;
        maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
        maxThreadsDim[0] = deviceProp.maxThreadsDim[0];
        maxThreadsDim[1] = deviceProp.maxThreadsDim[1];
        maxThreadsDim[2] = deviceProp.maxThreadsDim[2];
        maxThreadsDim[3] = deviceProp.maxThreadsDim[3];
        maxGridSize[0] = deviceProp.maxGridSize[0];
        maxGridSize[1] = deviceProp.maxGridSize[1];
        maxGridSize[2] = deviceProp.maxGridSize[2];
        maxGridSize[3] = deviceProp.maxGridSize[3];
        maxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;
        maxBlocksPerMultiProcessor = deviceProp.maxBlocksPerMultiProcessor;
    }
}

void CUDA::PrintInfo() {
    if (deviceCount == -1) {
        GetProperty();
    }

    printf("CUDA: deviceCount: %i\n", deviceCount);

    if (deviceCount == 0) {
        return;
    }

    printf("CUDA:                     nameGPU: %s\n", nameGPU.c_str());
    printf("CUDA:                    warpSize: %i\n", warpSize);
    printf("CUDA:          maxThreadsPerBlock: %i\n", maxThreadsPerBlock);
    printf("CUDA:            maxThreadsDim[0]: %i\n", maxThreadsDim[0]);
    printf("CUDA:            maxThreadsDim[1]: %i\n", maxThreadsDim[1]);
    printf("CUDA:            maxThreadsDim[2]: %i\n", maxThreadsDim[2]);
    printf("CUDA:            maxThreadsDim[3]: %i\n", maxThreadsDim[3]);
    printf("CUDA:              maxGridSize[0]: %i\n", maxGridSize[0]);
    printf("CUDA:              maxGridSize[1]: %i\n", maxGridSize[1]);
    printf("CUDA:              maxGridSize[2]: %i\n", maxGridSize[2]);
    printf("CUDA:              maxGridSize[3]: %i\n", maxGridSize[3]);
    printf("CUDA: maxThreadsPerMultiProcessor: %i\n", maxThreadsPerMultiProcessor);
    printf("CUDA:  maxBlocksPerMultiProcessor: %i\n", maxBlocksPerMultiProcessor);
}

namespace {
    void GetForceCPU(int count, int offset, float* masses, float* positionsX, float* positionsY, float* forcesX, float* forcesY, int threadId) {
        double _constGravity = 0.01f;
        int statIndex = threadId * offset;
        int endIndex = statIndex + offset;
        if (endIndex > count) {
            endIndex = count;
        }

        int sizeData = count;

        for (int index = statIndex; index < endIndex; ++index) {
            for (size_t otherIndex = 0; otherIndex < sizeData; ++otherIndex) {
                if (index == otherIndex) {
                    continue;
                }

                float gravityVecX = positionsX[otherIndex] - positionsX[index];
                float gravityVecY = positionsY[otherIndex] - positionsY[index];

                double dist = sqrt(gravityVecX * gravityVecX + gravityVecY * gravityVecY);
                gravityVecX /= dist;
                gravityVecY /= dist;

                double force = _constGravity * (masses[index] * masses[otherIndex]) / (dist * dist);
                gravityVecX *= force;
                gravityVecY *= force;

                forcesX[index] += gravityVecX;
                forcesY[index] += gravityVecY;
            }
        }
    }
}

__global__
void GetForceGPU(int* count, int* offset, float* masses, float* positionsX, float* positionsY, float* forcesX, float* forcesY) {
    double _constGravity = 0.01f;
    int statIndex = blockIdx.x * *offset;
    int endIndex = statIndex + *offset;
    if (endIndex > *count) {
        endIndex = *count;
    }

    int sizeData = *count;

    for (int index = statIndex; index < endIndex; ++index) {
        for (size_t otherIndex = 0; otherIndex < sizeData; ++otherIndex) {
            if (index == otherIndex) {
                continue;
            }

            float gravityVecX = positionsX[otherIndex] - positionsX[index];
            float gravityVecY = positionsY[otherIndex] - positionsY[index];

            double dist = sqrt(gravityVecX * gravityVecX + gravityVecY * gravityVecY);
            gravityVecX /= dist;
            gravityVecY /= dist;

            double force = _constGravity * (masses[index] * masses[otherIndex]) / (dist * dist);
            gravityVecX *= force;
            gravityVecY *= force;

            forcesX[index] += gravityVecX;
            forcesY[index] += gravityVecY;
        }
    }
}

void CUDA::GetForcesStaticTest(int count, float* masses, float* positionsX, float* positionsY, float* forcesX, float* forcesY) {
    unsigned int counThread = static_cast<double>(std::thread::hardware_concurrency());
    unsigned int sizeB = count;

    if ((sizeB * 2) > counThread) {
        int offst = sizeB / counThread;
        if ((sizeB % counThread) > 0) {
            ++offst;
        }

        std::vector<std::thread> threads;
        threads.reserve(counThread);

        for (unsigned int threadId = 0; threadId < counThread; ++threadId) {
            threads.emplace_back([&]() {
                GetForceCPU(count, offst, masses, positionsX, positionsY, forcesX, forcesY, threadId);
            });
        }

        for (std::thread& th : threads) {
            th.join();
        }
    }
    else
    {
        GetForceCPU(count, count, masses, positionsX, positionsY, forcesX, forcesY, 0);
    }
}

void CUDA::GetForcesStatic(int count, float* masses, float* positionsX, float* positionsY, float* forcesX, float* forcesY) {
    unsigned int counThread = CUDA::warpSize;
    if (counThread == 0) {
        return;
    }

    int offst = count / counThread;
    if ((count % counThread) > 0) {
        ++offst;
    }

    //...
    int* devCount;
    int* devOffset;
    float* devMasses;
    float* devPositionsX;
    float* devPositionsY;
    float* devForcesX;
    float* devForcesY;

    cudaMalloc(&devCount,               sizeof(float));
    cudaMalloc(&devOffset,              sizeof(float));
    cudaMalloc(&devMasses,      count * sizeof(float));
    cudaMalloc(&devPositionsX,  count * sizeof(float));
    cudaMalloc(&devPositionsY,  count * sizeof(float));
    cudaMalloc(&devForcesX,     count * sizeof(float));
    cudaMalloc(&devForcesY,     count * sizeof(float));

    cudaMemcpy(devCount,        &count,     count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devOffset,       &offst,     count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devMasses,       masses,     count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devPositionsX,   positionsX, count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devPositionsY,   positionsY, count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devForcesX,      forcesX,    count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devForcesY,      forcesY,    count * sizeof(float), cudaMemcpyHostToDevice);

    GetForceGPU <<<counThread, 1>>> (devCount, devOffset, devMasses, devPositionsX, devPositionsY, devForcesX, devForcesY);

    cudaMemcpy(devForcesX, forcesX, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(devForcesY, forcesY, count * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(devCount);
    cudaFree(devOffset);
    cudaFree(devMasses);
    cudaFree(positionsX);
    cudaFree(positionsY);
    cudaFree(devForcesX);
    cudaFree(devForcesY);
}

//...
#else
    void testCUDA(void) {}

    void CUDA::GetProperty() {}
    void CUDA::PrintInfo() {}
#endif
