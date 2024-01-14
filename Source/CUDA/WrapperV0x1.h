#pragma once
#include <vector>
#include "Wrapper.h"

class WrapperV0x1 final {
public:
	static void UpdatePositionGPU(unsigned int count, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces, CUDA::Vector3* velocities);
	static void UpdatePositionCPU(unsigned int count, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces, CUDA::Vector3* velocities);

public:
	static int tag;
	static int tagCurrent;
};
