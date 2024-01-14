#include "Emulate.h"
#include "Wrapper.h"
#include <stdio.h>
#include <vector>
#include <thread>

#if ENABLE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace CUDA_TEST {
	CUDA_Emulate::CUDA_Emulate(unsigned int countBlock, unsigned int countThread, std::function<void(CUDA_TEST::Index)> fun) {
		std::vector<std::thread> threads;
		threads.reserve(countBlock * countThread);

		for (unsigned int iThread = 0; iThread < countThread; ++iThread) {
			for (unsigned int iBlock = 0; iBlock < countBlock; ++iBlock) {
				threads.emplace_back([iThread, iBlock, countThread, fun]() {
					fun(CUDA_TEST::Index(iBlock, iThread, countThread));
				});
			}
		}

		for (std::thread& th : threads) {
			th.join();
		}
	}

	unsigned int CUDA_Emulate::cudaMemcpyHostToDevice = 1;
	unsigned int CUDA_Emulate::cudaMemcpyDeviceToHost = 2;
}

/// CPU //////////////////////////////////////////////////////////////////////////////

namespace {
	void UpdateCPU(unsigned int* count, unsigned int* offset, CUDA::Vector3* positions, CUDA_TEST::Index indexData) {
		int indexT = indexData.threadIdx.x + indexData.blockIdx.x * indexData.blockDim.x;
		int startIndex = indexT * *offset;
		int countIndex = startIndex + *offset;
		if (countIndex >= *count) {
			countIndex = *count;
		}
		//printf("\nUpdateCpu INDEXES [%i, %i] [%i, %i, %i]\n", startIndex, countIndex, indexData.threadIdx.x, indexData.blockIdx.x, indexData.blockDim.x);

		for (int index = startIndex; index < countIndex; ++index) {
			CUDA::Vector3& pos = positions[index];
			//printf("          POS: [%i] [%f, %f, %f] => ", index, pos.x, pos.y, pos.z);
			pos.x += pos.x;
			pos.y += pos.y;
			pos.z += pos.z;
			//printf(" [%f, %f, %f]\n", pos.x, pos.y, pos.z);
		}
	}
}

/// GPU //////////////////////////////////////////////////////////////////////////////

namespace {
	__global__ void UpdateGPU(unsigned int* count, unsigned int* offset, CUDA::Vector3* positions) {
		int indexT = threadIdx.x + blockIdx.x * blockDim.x;
		int startIndex = indexT * *offset;
		int countIndex = startIndex + *offset;
		if (countIndex >= *count) {
			countIndex = *count;
		}
		//printf("\nUpdateGpu INDEXES [%i, %i] [%i, %i, %i]\n", startIndex, countIndex, threadIdx.x, blockIdx.x, blockDim.x);

		for (int index = startIndex; index < countIndex; ++index) {
			CUDA::Vector3& pos = positions[index];
			//printf("          POS: [%i] [%f, %f, %f] => ", index, pos.x, pos.y, pos.z);
			pos.x += pos.x;
			pos.y += pos.y;
			pos.z += pos.z;
			//printf(" [%f, %f, %f]\n", pos.x, pos.y, pos.z);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////

void CUDA_TEST::Test() {
	unsigned int count = 100;
	
	std::vector<float> masses;
	masses.reserve(count);
	std::vector<CUDA::Vector3> positionsCpu;
	positionsCpu.reserve(count);
	std::vector<CUDA::Vector3> positionsGpu;
	positionsGpu.reserve(count);

	for (int i = (count-1); i >= 0; --i) {
		positionsCpu.emplace_back((float)i, (float)i, (float)i);
		positionsGpu.emplace_back((float)i, (float)i, (float)i);
		masses.emplace_back((float)i + 1.f);
	}

	// CPU
	{
		std::vector<CUDA::Vector3>& positions = positionsCpu;
		std::vector<CUDA::Vector3> forces;
		forces.resize(count);

		const unsigned int maxCountBlock = 16;
		const unsigned int maxCountThread = 16;

		unsigned int countBlock;
		unsigned int countThread;
		unsigned int offset;

		CUDA::GetOffsets(count, maxCountBlock, maxCountThread, countBlock, countThread, offset);
		printf("\nCPU BEGIN count: %i, offset %i, countBlock %i, countThread %i\n", count, offset, countBlock, countThread);

		unsigned int* devCount;
		unsigned int* devOffset;
		CUDA::Vector3* devPositions;
		float* devMasses;
		CUDA::Vector3* devForces;

		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devCount, sizeof(unsigned int));
		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devOffset, sizeof(unsigned int));
		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devPositions, count * sizeof(CUDA::Vector3));
		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devMasses, count * sizeof(float));
		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devForces, count * sizeof(CUDA::Vector3));

		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devCount, &count, sizeof(unsigned int), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devOffset, &offset, sizeof(unsigned int), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devPositions, positions.data(), count * sizeof(CUDA::Vector3), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devMasses, masses.data(), count * sizeof(float), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devForces, forces.data(), count * sizeof(CUDA::Vector3), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);

		CUDA_Emulate(countBlock, countThread, [devCount, devOffset, devPositions](Index indexData) {
			UpdateCPU(devCount, devOffset, devPositions, indexData);
		});

		CUDA_TEST::CUDA_Emulate::cudaMemcpy(&count, devCount, sizeof(unsigned int), CUDA_TEST::CUDA_Emulate::cudaMemcpyDeviceToHost);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(&offset, devOffset, sizeof(unsigned int), CUDA_TEST::CUDA_Emulate::cudaMemcpyDeviceToHost);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(positions.data(), devPositions, count * sizeof(CUDA::Vector3), CUDA_TEST::CUDA_Emulate::cudaMemcpyDeviceToHost);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(masses.data(), devMasses, count * sizeof(float), CUDA_TEST::CUDA_Emulate::cudaMemcpyDeviceToHost);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(forces.data(), devForces, count * sizeof(CUDA::Vector3), CUDA_TEST::CUDA_Emulate::cudaMemcpyDeviceToHost);

		CUDA_TEST::CUDA_Emulate::cudaFree(devCount);
		CUDA_TEST::CUDA_Emulate::cudaFree(devOffset);
		CUDA_TEST::CUDA_Emulate::cudaFree(devPositions);
		CUDA_TEST::CUDA_Emulate::cudaFree(devMasses);
		CUDA_TEST::CUDA_Emulate::cudaFree(devForces);

		printf("\nCPU END \n");
	}

	// GPU
	{
		std::vector<CUDA::Vector3>& positions = positionsGpu;
		std::vector<CUDA::Vector3> forces;
		forces.resize(count);

		const unsigned int maxCountBlock = 16;
		const unsigned int maxCountThread = 16;

		unsigned int countBlock;
		unsigned int countThread;
		unsigned int offset;

		CUDA::GetOffsets(count, maxCountBlock, maxCountThread, countBlock, countThread, offset);
		printf("\nGPU BEGIN count: %i, offset %i, countBlock %i, countThread %i\n", count, offset, countBlock, countThread);

		unsigned int* devCount;
		unsigned int* devOffset;
		CUDA::Vector3* devPositions;
		float* devMasses;
		CUDA::Vector3* devForces;

		cudaMalloc(&devCount, sizeof(unsigned int));
		cudaMalloc(&devOffset, sizeof(unsigned int));
		cudaMalloc(&devPositions, count * sizeof(CUDA::Vector3));
		cudaMalloc(&devMasses, count * sizeof(float));
		cudaMalloc(&devForces, count * sizeof(CUDA::Vector3));

		cudaMemcpy(devCount, &count, sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(devOffset, &offset, sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(devPositions, positions.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);
		cudaMemcpy(devMasses, masses.data(), count * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(devForces, forces.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);

		UpdateGPU << <countBlock, countThread >> > (devCount, devOffset, devPositions);

		cudaMemcpy(&count, devCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&offset, devOffset, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(positions.data(), devPositions, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);
		cudaMemcpy(masses.data(), devMasses, count * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(forces.data(), devForces, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);

		cudaFree(devCount);
		cudaFree(devOffset);
		cudaFree(devPositions);
		cudaFree(devMasses);
		cudaFree(devForces);

		printf("\nGPU END \n");
	}

	bool equal = true;
	bool printData = true;;
	{
		for (size_t i = 0; i < count; ++i) {
			if (positionsCpu[i].x == positionsGpu[i].x && positionsCpu[i].y == positionsGpu[i].y && positionsCpu[i].z == positionsGpu[i].z) {
				if (printData) {
					printf("\tPOS[%i]:\n[%f, %f, %f] !=\n[%f, %f, %f]\n", i,
						positionsCpu[i].x, positionsCpu[i].y, positionsCpu[i].z,
						positionsGpu[i].x, positionsGpu[i].y, positionsGpu[i].z);
				}
			} else {
				equal = false;

				if (printData) {
					printf("\tPOS[%i]:\n[%f, %f, %f] !=\n[%f, %f, %f] FAIL\n", i,
						positionsCpu[i].x, positionsCpu[i].y, positionsCpu[i].z,
						positionsGpu[i].x, positionsGpu[i].y, positionsGpu[i].z);
				} else {
					break;
				}
			}
		}
	}

	if (equal) {
		printf("\nCOMPARE OK \n...............................................\n");
	} else {
		printf("\nCOMPARE FAIL \n...............................................\n");
	}	
}

#else

void CUDA_TEST::Test() { }

#endif
