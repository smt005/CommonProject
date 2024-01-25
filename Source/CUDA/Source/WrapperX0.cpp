
#include "WrapperX0.h"

bool WrapperX0::sync = false;
int WrapperX0::tag = 0;
int WrapperX0::tagCurrent = -1;

void WrapperX0::UpdatePositionGPU(unsigned int count, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces, CUDA::Vector3* velocities, float dt, unsigned int countOfIteration) { }
void WrapperX0::UpdatePositionCPU(unsigned int count, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces, CUDA::Vector3* velocities, float dt, unsigned int countOfIteration) { }
