#pragma once
#include <vector>
#include "Wrapper.h"

class CUDA_PrototypeV3 final {
public:
	static void GetForcesCPUStatic(std::vector<CUDA::Body>& bodies, std::vector<CUDA::Vector3>& forces);
	static void GetForcesGPUStatic(std::vector<CUDA::Body>& bodies, std::vector<CUDA::Vector3>& forces);

public:
	static int tag;
	static int tagCurrent;
};
