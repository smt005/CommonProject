#pragma once

class CUDA_PrototypeV3 final {
public:
	static void GetForcesCPUStatic(int count, float* masses, float* positionsX, float* positionsY, float* forcesX, float* forcesY);
	static void GetForcesGPUStatic(int count, float* masses, float* positionsX, float* positionsY, float* forcesX, float* forcesY);

public:
	static int tag;
	static int tagCurrent;
};
