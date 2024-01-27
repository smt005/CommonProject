#pragma once
#include <vector>
#include "Wrapper.h"
#include "Classes.h"

class WrapperX1 final {
public:
	static void CalculateForceCPU(cuda::Buffer& buffer);
	static void UpdatePositionCPU(cuda::Buffer& buffer, float dt);

	static void CalculateForceGPU(cuda::Buffer& buffer);
	static void UpdatePositionGPU(cuda::Buffer& buffer, float dt);

public:
	static bool sync;
	static int tag;
	static int tagCurrent;
};
