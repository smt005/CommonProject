
#include "WrapperX1.h"

bool WrapperX1::sync = false;
int WrapperX1::tag = 0;
int WrapperX1::tagCurrent = -1;

void WrapperX1::UpdatePositionGPU(unsigned int count, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces, CUDA::Vector3* velocities, float dt, unsigned int countOfIteration) { }
void WrapperX1::UpdatePositionCPU(unsigned int count, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces, CUDA::Vector3* velocities, float dt, unsigned int countOfIteration) { }
