
#include "Wrapper.h"
#include <thread>
#include <vector>

bool CUDA::processGPU = false;
bool CUDA::multithread = false;

std::string CUDA::nameGPU;
int         CUDA::deviceCount = -1;
int         CUDA::warpSize = 0;
int         CUDA::maxThreadsPerBlock = 0;
int         CUDA::maxThreadsDim[3];
int         CUDA::maxGridSize[3];
int         CUDA::maxThreadsPerMultiProcessor = 0;
int         CUDA::maxBlocksPerMultiProcessor = 0;

CUDA::Vector3::Vector3()
    : x(0.0)
    , y(0.0)
    , z(0.0)
{}

CUDA::Vector3::Vector3(float _x, float _y, float _z)
    : x(_x)
    , y(_y)
    , z(_z)
{}

CUDA::Body::Body(float _posX, float _posY, float _posZ, float _mass, float _velX, float _velY, float _velZ)
    : pos(_posX, _posY, _posZ)
    , mass(_mass)
{}

void CUDA::GetProperty() {}
void CUDA::PrintInfo() {}

void CUDA::GetForcesStaticTest(int count, float* masses, float* positionsX, float* positionsY, float* forcesX, float* forcesY) {}
void CUDA::GetForcesStatic(int count, float* masses, float* positionsX, float* positionsY, float* forcesX, float* forcesY) {}

void CUDA::GetOffsets(const unsigned int count, const unsigned int maxBlocks, const unsigned int maxThreads, unsigned int& countBlock, unsigned int& countThread, unsigned int& offset) {}
