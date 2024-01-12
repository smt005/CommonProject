#pragma once

#include <string>

class CUDA final {
public:
	struct Vector3 {
		Vector3();
		Vector3(float _x, float _y, float _z);
		float x, y, z;
	};

	struct Body {
		Body(float _posX, float _posY, float _posZ, float _mass, float _velX, float _velY, float _velZ);
		Vector3 pos;
		float mass;
	};

public:
	static void GetProperty();
	static void PrintInfo();

	static void GetForcesStaticTest(int count, float* masses, float* positionsX, float* positionsY, float* forcesX, float* forcesY);
	static void GetForcesStatic(int count, float* masses, float* positionsX, float* positionsY, float* forcesX, float* forcesY);

public:
	static std::string	nameGPU;
	static int			deviceCount;
	static int          warpSize;                   /**< Warp size in threads */
	static int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
	static int          maxThreadsDim[3];           /**< Maximum size of each dimension of a block */
	static int          maxGridSize[3];             /**< Maximum size of each dimension of a grid */
	static int          maxThreadsPerMultiProcessor;/**< Maximum resident threads per multiprocessor */
	static int          maxBlocksPerMultiProcessor; /**< Maximum number of resident blocks per multiprocessor */
};
