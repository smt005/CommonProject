#pragma once
#include <vector>
#include "Wrapper.h"

class WrapperX0 final {
public:
	static void UpdatePositionGPU(unsigned int count, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces, CUDA::Vector3* velocities, float dt, unsigned int countOfIteration);
	static void UpdatePositionCPU(unsigned int count, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces, CUDA::Vector3* velocities, float dt, unsigned int countOfIteration);

public:
	static bool sync;
	static int tag;
	static int tagCurrent;
};
