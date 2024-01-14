#include "Test.h"

#if ENABLE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <stdio.h>
#include <algorithm>

#include "Wrapper.h"

namespace CUDA_TEST {
	// threadIdx.x
	struct ThreadIdx {
		unsigned int x = 0;
	} _threadIdx;

	// blockIdx.x
	struct BlockIdx {
		unsigned int x = 0;
	} _blockIdx;

	// blockDim.x
	struct BlockDim {
		unsigned int x = 0;
	} _blockDim;

	template<unsigned int countBlock, unsigned int countThread>
	void UmulateCuda(std::function<void(void)> fun) {
		_blockDim.x = countThread;

		for (unsigned int iBlock = 0; iBlock < countBlock; ++iBlock) {
			_blockIdx.x = iBlock;

			for (unsigned int iThread = 0; iThread < countThread; ++iThread) {
				_threadIdx.x = iThread;

				fun();
			}
		}			
	}

	double Time() {
		std::chrono::milliseconds ms;
		ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
		return static_cast<double>(ms.count());
	}

	// Functions
	void CalcForcesCpu(int* count, int* offset, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces) {
		int indexT = _threadIdx.x + _blockIdx.x * _blockDim.x;
		int startIndex = indexT * *offset;
		int sizeIndex = startIndex + *offset;
		if (sizeIndex >= *count) {
			sizeIndex = *count;
		}

		float constGravity = 0.01;

		float gravityX;
		float gravityY;
		float gravityZ;
		float dist = 0;
		float force = 0;

		// INFO
		/*printf("CalcForcesCpu statIndex: %i sizeIndex: %i\n", startIndex, sizeIndex);
		for (int index = startIndex; index < sizeIndex; ++index) {
			printf("CalcForcesCpu positions: [%i] [%f, %f, %f], masses: %f, forces: [%f, %f, %f]\n",
				index,
				positions[index].x, positions[index].y, positions[index].z,
				masses[index],
				forces[index].x, forces[index].y, forces[index].z);
		}*/

		for (int index = startIndex; index < sizeIndex; ++index) {
			CUDA::Vector3* pos = &positions[index];
			float mass = masses[index];
			forces[index].x = 0.f;
			forces[index].y = 0.f;
			forces[index].z = 0.f;

			for (size_t otherIndex = 0; otherIndex < *count; ++otherIndex) {
				if (index == otherIndex) {
					continue;
				}

				gravityX = positions[otherIndex].x - positions[index].x;
				gravityY = positions[otherIndex].y - positions[index].y;
				gravityZ = positions[otherIndex].z - positions[index].z;

				dist = sqrt(gravityX * gravityX + gravityY * gravityY + gravityZ * gravityZ);
				gravityX /= dist;
				gravityY /= dist;
				gravityZ /= dist;

				force = constGravity * (mass * masses[otherIndex]) / (dist * dist);
				gravityX *= force;
				gravityY *= force;
				gravityZ *= force;

				forces[index].x += gravityX;
				forces[index].y += gravityY;
				forces[index].z += gravityZ;
			}
		}
	}

	void UpdatePositionsCpu(int* count, int* offset, CUDA::Vector3* positions, CUDA::Vector3* velocities, float* masses, CUDA::Vector3* forces, float* dt) {
		int indexT = _threadIdx.x + _blockIdx.x * _blockDim.x;
		int startIndex = indexT * *offset;
		int sizeIndex = startIndex + *offset;
		if (sizeIndex >= *count) {
			sizeIndex = *count;
		}

		float accelerationX;
		float accelerationY;
		float accelerationZ;
		float appendVelocityX;
		float appendVelocityY;
		float appendVelocityZ;

		// INFO
		/*printf("UpdatePositionsCpu statIndex: %i countIndex: %i, dt: %f\n", startIndex, sizeIndex, *dt);
		for (int index = startIndex; index < sizeIndex; ++index) {
			printf("UpdatePositionsCpu positions: [%i] [%f, %f, %f], velocities: [%f, %f, %f], masses: %f, forces: [%f, %f, %f]\n",
				index,
				positions[index].x, positions[index].y, positions[index].z,
				velocities[index].x, velocities[index].y, velocities[index].z,
				masses[index],
				forces[index].x, forces[index].y, forces[index].z);
		}*/

		for (int index = startIndex; index < sizeIndex; ++index) {
			accelerationX = forces[index].x / masses[index];
			accelerationY = forces[index].y / masses[index];
			accelerationZ = forces[index].z / masses[index];

			appendVelocityX = accelerationX * *dt;
			appendVelocityY = accelerationY * *dt;
			appendVelocityZ = accelerationZ * *dt;

			velocities[index].x += appendVelocityX;
			velocities[index].y += appendVelocityY;
			velocities[index].z += appendVelocityZ;

			positions[index].x = velocities[index].x;
			positions[index].y = velocities[index].y;
			positions[index].z = velocities[index].z;
		}
	}

	__global__ void CalcForcesGpu(int* count, int* offset, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces) {
		int indexT = threadIdx.x + blockIdx.x * blockDim.x;
		int startIndex = indexT * *offset;
		int sizeIndex = startIndex + *offset;
		if (sizeIndex >= *count) {
			sizeIndex = *count;
		}

		float constGravity = 0.01;

		float gravityX;
		float gravityY;
		float gravityZ;
		float dist = 0;
		float force = 0;

		// INFO
		/*printf("CalcForcesGpu statIndex: %i sizeIndex: %i\n", startIndex, sizeIndex);
		for (int index = startIndex; index < sizeIndex; ++index) {
			printf("CalcForcesGpu positions: [%i] [%f, %f, %f], masses: %f, forces: [%f, %f, %f]\n",
				index,
				positions[index].x, positions[index].y, positions[index].z,
				masses[index],
				forces[index].x, forces[index].y, forces[index].z);
		}*/

		for (int index = startIndex; index < sizeIndex; ++index) {
			CUDA::Vector3* pos = &positions[index];
			float mass = masses[index];
			forces[index].x = 0.f;
			forces[index].y = 0.f;
			forces[index].z = 0.f;

			for (size_t otherIndex = 0; otherIndex < *count; ++otherIndex) {
				if (index == otherIndex) {
					continue;
				}

				gravityX = positions[otherIndex].x - positions[index].x;
				gravityY = positions[otherIndex].y - positions[index].y;
				gravityZ = positions[otherIndex].z - positions[index].z;

				dist = sqrt(gravityX * gravityX + gravityY * gravityY + gravityZ * gravityZ);
				gravityX /= dist;
				gravityY /= dist;
				gravityZ /= dist;

				force = constGravity * (mass * masses[otherIndex]) / (dist * dist);
				gravityX *= force;
				gravityY *= force;
				gravityZ *= force;

				//forces[index].x += gravityX;
				//forces[index].y += gravityY;
				//forces[index].z += gravityZ;

				atomicAdd(&forces[index].x, gravityX);
				atomicAdd(&forces[index].y, gravityY);
				atomicAdd(&forces[index].z, gravityZ);
			}
		}
	}

	__global__ void UpdatePositionsGpu(int* count, int* offset, CUDA::Vector3* positions, CUDA::Vector3* velocities, float* masses, CUDA::Vector3* forces, float* dt) {
		int indexT = threadIdx.x + blockIdx.x * blockDim.x;
		int startIndex = indexT * *offset;
		int sizeIndex = startIndex + *offset;
		if (sizeIndex >= *count) {
			sizeIndex = *count;
		}

		float accelerationX;
		float accelerationY;
		float accelerationZ;
		float appendVelocityX;
		float appendVelocityY;
		float appendVelocityZ;

		// INFO
		/*printf("UpdatePositionsGpu statIndex: %i countIndex: %i, dt: %f\n", startIndex, sizeIndex, *dt);
		for (int index = startIndex; index < sizeIndex; ++index) {
			printf("UpdatePositionsGpu positions: [%i] [%f, %f, %f], velocities: [%f, %f, %f], masses: %f, forces: [%f, %f, %f]\n",
				index,
				positions[index].x, positions[index].y, positions[index].z,
				velocities[index].x, velocities[index].y, velocities[index].z,
				masses[index],
				forces[index].x, forces[index].y, forces[index].z);
		}*/

		for (int index = startIndex; index < sizeIndex; ++index) {
			accelerationX = forces[index].x / masses[index];
			accelerationY = forces[index].y / masses[index];
			accelerationZ = forces[index].z / masses[index];

			appendVelocityX = accelerationX * *dt;
			appendVelocityY = accelerationY * *dt;
			appendVelocityZ = accelerationZ * *dt;

			velocities[index].x += appendVelocityX;
			velocities[index].y += appendVelocityY;
			velocities[index].z += appendVelocityZ;

			positions[index].x = velocities[index].x;
			positions[index].y = velocities[index].y;
			positions[index].z = velocities[index].z;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// CUDA_Test::Run //////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CUDA_Test::Run() {
	constexpr int count = 10000;// 1048576; // 512 1024 1048576
	
	std::vector<CUDA::Vector3> positions;
	std::vector<float> masses;
	std::vector<CUDA::Vector3> velocities;
	float dt = 1.f;
	int reserveCount = count;

	printf("Test::Run begin count: %i.\n", count);

	if (positions.empty()) {
		positions.reserve(count);
		masses.reserve(count);
		velocities.reserve(count);

		for (int i = 0; i < count; ++i) {
			positions.emplace_back((float)i, (float)i, (float)i);
			masses.emplace_back((float)i + 1.f);
			velocities.emplace_back((float)i, (float)i, (float)i);
		}
	}

	// CPU
	std::vector<CUDA::Vector3> cpuPositions;
	{
		constexpr int maxCountBlock = 11;
		constexpr int maxCountThread = 16;

		constexpr int countThread = maxCountThread;
		constexpr int countBlock = ((count + countThread - 1) / countThread) > maxCountBlock ? maxCountBlock : ((count + countThread - 1) / countThread);
		constexpr int offset = (count + (countBlock * countThread) - 1) / (countBlock * countThread);

		cpuPositions = positions;
		std::vector<CUDA::Vector3> cpuVelocities = velocities;

		printf("Test::Run CPU begin count: %i, offset: %i, blocks: %i, threads: %i.\n", count, offset, countBlock, countThread);
		auto timeAll = CUDA_TEST::Time();

		auto timeCpy = CUDA_TEST::Time();
		std::vector<CUDA::Vector3> forces;
		forces.resize(count, CUDA::Vector3());

		int* devCount = new int(count);
		int* devOffset = new int(offset);
		timeCpy = CUDA_TEST::Time() - timeCpy;

		auto time = CUDA_TEST::Time();
		auto time0 = time;
		CUDA_TEST::UmulateCuda<countBlock, countThread>([devCount, devOffset, devPositions = cpuPositions.data(), devMasses = masses.data(), devForces = forces.data()]() {
			CUDA_TEST::CalcForcesCpu(devCount, devOffset, devPositions, devMasses, devForces);
		});
		time0 = CUDA_TEST::Time() - time0;

		auto time1 = CUDA_TEST::Time();
		CUDA_TEST::UmulateCuda<countBlock, countThread>([devCount, devOffset, devPositions = cpuPositions.data(), devVelocities = cpuVelocities.data(), devMasses = masses.data(), devForces = forces.data(), &dt]() {
			CUDA_TEST::UpdatePositionsCpu(devCount, devOffset, devPositions, devVelocities, devMasses, devForces, &dt);
		});
		time1 = CUDA_TEST::Time() - time1;
		time = CUDA_TEST::Time() - time;

		delete devCount;
		delete devOffset;

		timeAll = CUDA_TEST::Time() - timeAll;
		printf("Test::Run CPU: [[%f(%f), %f(%f)], %f(%f), !%f(%f)!], %f ms] end\n\n", time0, (time0/16), time1, (time1 / 16), time, (time / 16), timeAll, (timeAll / 16),
			timeCpy);
	}

	// GPU
	std::vector<CUDA::Vector3> gpuPositions;
	{
		int maxCountBlock = CUDA::maxGridSize[1];
		int maxCountThread = CUDA::maxThreadsPerBlock;

		int countThread = maxCountThread;
		int countBlock = ((count + countThread - 1) / countThread) > maxCountBlock ? maxCountBlock : ((count + countThread - 1) / countThread);
		int offset = (count + (countBlock * countThread) - 1) / (countBlock * countThread);

		gpuPositions.resize(count);
		std::vector<CUDA::Vector3> gpuVelocities = velocities;

		printf("Test::Run GPU begin count: %i, offset: %i, blocks: %i, threads: %i.\n", count, offset, countBlock, countThread);
		auto timeAll = CUDA_TEST::Time();

		int* devCount;
		int* devOffset;
		float* devDt;
		CUDA::Vector3* devPositions;
		float* devMasses;
		CUDA::Vector3* devVelocities;
		CUDA::Vector3* devForces;

		auto timeMem = CUDA_TEST::Time();
		cudaMalloc(&devCount,	sizeof(int));
		cudaMalloc(&devOffset, sizeof(int));
		cudaMalloc(&devDt,	sizeof(float));
		cudaMalloc(&devPositions,	count * sizeof(CUDA::Vector3));
		cudaMalloc(&devMasses,	count * sizeof(float));
		cudaMalloc(&devVelocities,	count * sizeof(CUDA::Vector3));
		cudaMalloc(&devForces, count * sizeof(CUDA::Vector3));
		timeMem = CUDA_TEST::Time() - timeMem;
		//printf("CUDA cudaMalloc: %f ms\n", timeMem);

		auto timeCpyToDev = CUDA_TEST::Time();
		cudaMemcpy(devCount,	&count, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devOffset, &offset, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devDt,	&dt, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(devPositions, positions.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);
		cudaMemcpy(devMasses, masses.data(), count * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(devVelocities, gpuVelocities.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);
		timeCpyToDev = CUDA_TEST::Time() - timeCpyToDev;
		//printf("CUDA to dev  cudaMemcpy: %f ms\n", timeCpyToDev);

		auto time = CUDA_TEST::Time();
		auto time0 = time;
		CUDA_TEST::CalcForcesGpu << <countBlock, countThread >> > (devCount, devOffset, devPositions, devMasses, devForces);
		time0 = CUDA_TEST::Time() - time0;

		auto time1 = CUDA_TEST::Time();
		CUDA_TEST::UpdatePositionsGpu << <countBlock, countThread >> > (devCount, devOffset, devPositions, devVelocities, devMasses, devForces, devDt);
		time1 = CUDA_TEST::Time() - time1;
		time = CUDA_TEST::Time() - time;
		//printf("CUDA fun: %f ms\n", time);

		auto timeCpyToHost = CUDA_TEST::Time();
		cudaMemcpy(gpuPositions.data(), devPositions, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);
		cudaMemcpy(gpuVelocities.data(), devVelocities, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);
		timeCpyToHost = CUDA_TEST::Time() - timeCpyToHost;
		//printf("CUDA to host cudaMemcpy: %f ms\n", timeCpyToHost);

		auto timeFree = CUDA_TEST::Time();
		cudaFree(devCount);
		cudaFree(devDt);
		cudaFree(devPositions);
		cudaFree(devMasses);
		cudaFree(devForces);
		cudaFree(devVelocities);
		timeFree = CUDA_TEST::Time() - timeFree;
		//printf("CUDA cudaFree: %f ms\n", timeFree);

		timeAll = CUDA_TEST::Time() - timeAll;
		printf("Test::Run GPU: [[%f, %f], %f, !%f!], [%f, %f, %f, %f] ms] end\n\n",
			time0, time1, time, timeAll,
			timeMem, timeCpyToDev, timeCpyToHost, timeFree);
	}

	printf("\nTest::Run end.\n");

	auto compare = [](auto left, auto right) {
		auto val = std::abs(left - right);
		//return val < 0.000001f;
		return val == 0.0;// 0000001f;
	};

	//...
	bool equal = true;
	for (size_t i = 0; i < count; ++i) {
		if (!(compare(cpuPositions[i].x, gpuPositions[i].x) && compare(cpuPositions[i].y, gpuPositions[i].y) && compare(cpuPositions[i].z, gpuPositions[i].z))) {
			equal = false;
			break;
		}
	}
	
	if (equal) {
		printf("Test::Run result OK.\n\n\n");
	}
	else {
		printf("Test::Run result FAIL.\n\n\n");
	}

	/*for (size_t i = 0; i < count; ++i) {
		if (!(compare(cpuPositions[i].x, gpuPositions[i].x) && compare(cpuPositions[i].y, gpuPositions[i].y) && compare(cpuPositions[i].z, gpuPositions[i].z))) {
			printf("\tpos[%i]:\n[%f, %f, %f] !=\n[%f, %f, %f] FAIL\n", i,
				cpuPositions[i].x, cpuPositions[i].y, cpuPositions[i].z,
				gpuPositions[i].x, gpuPositions[i].y, gpuPositions[i].z);
		}
		
	}*/
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// RunTestIndex ////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace CUDA_TEST {

void TestIndexForCpu(int* count, int* offset, int* indexes, int* result) {
	int indexT = _threadIdx.x + _blockIdx.x * _blockDim.x;
	printf("\nTestIndexForCpu APPEND index: %i (%i, %i, %i)\n", indexT, _threadIdx.x, _blockIdx.x, _blockDim.x);

	int startIndex = indexT * *offset;
	int sizeIndex = startIndex + *offset;
	if (sizeIndex >= *count) {
		sizeIndex = *count;
		printf("TestIndexForCpu CORRECT [%i, %i]\n", sizeIndex, *count);
	}

	for (int index = startIndex; index < sizeIndex; ++index) {
		indexes[index] = index;
		*result += index;

		printf("TestIndexForCpu [%i] APPEND index: %i\n", indexT, index);
	}
}

__global__ void TestIndexForGpu(int* count, int* offset, int* indexes, int* result) {
	int indexT = threadIdx.x + blockIdx.x * blockDim.x;
	printf("\nTestIndexForGpu APPEND index: %i (%i, %i, %i)\n", indexT, threadIdx.x, blockIdx.x, blockDim.x);

	int startIndex = indexT * *offset;
	int sizeIndex = startIndex + *offset;
	if (sizeIndex >= *count) {
		sizeIndex = *count;
		printf("TestIndexForGpu CORRECT [%i, %i]\n", sizeIndex, *count);
	}

	for (int index = startIndex; index < sizeIndex; ++index) {
		int indexTemp = 0;

		if (index < *count) {
			indexes[index] = index;
			indexTemp = index;
			printf("TestIndexForGpu [%i] APPEND index: %i\n", indexT, index);
		}

		//*result = *result + indexTemp; // Потоконебезопасно
		atomicAdd(result, indexTemp);
	}
}

}

void CUDA_Test::RunTestIndex() {
	printf("\nTest::RunTestIndex BEGIN.\n");

	constexpr int count = 55;
	int reserveCount = count + 10;
	constexpr int maxCountBlock = 3;
	constexpr int maxCountThread = 2;

	constexpr int countThread = maxCountThread;
	constexpr int countBlock = ((count + countThread - 1) / countThread) > maxCountBlock ? maxCountBlock : ((count + countThread - 1) / countThread);
	constexpr int offset = (count + (countBlock * countThread) - 1) / (countBlock * countThread);

	// CPU
	int resultCpu = 0;
	std::vector<int> indexesCpu;
	{
		printf("\nCPU .   .   .\n");

		indexesCpu.resize(reserveCount, std::numeric_limits<int>::max());

		int* devCount = new int(count);
		int* devOffset = new int(offset);

		CUDA_TEST::UmulateCuda<countBlock, countThread>([devCount, devOffset, devResult = &resultCpu, devIndexes = indexesCpu.data()]() {
			CUDA_TEST::TestIndexForCpu(devCount, devOffset, devIndexes, devResult);
		});

		delete devCount;
		delete devOffset;
	}

	// GPU
	int resultGpu = 0;
	std::vector<int> indexesGpu;
	{
		printf("\nGPU .   .   .\n");
		
		indexesGpu.resize(reserveCount, std::numeric_limits<int>::max());

		int* devCount;
		int* devOffset;
		int* devResult;
		int* devIndexes;

		cudaMalloc(&devCount, sizeof(int));
		cudaMalloc(&devOffset, sizeof(int));
		cudaMalloc(&devResult, sizeof(int));
		cudaMalloc(&devIndexes, reserveCount * sizeof(int));

		cudaMemcpy(devCount, &count, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devOffset, &offset, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devResult, &resultGpu, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devIndexes, indexesGpu.data(), reserveCount * sizeof(int), cudaMemcpyHostToDevice);

		CUDA_TEST::TestIndexForGpu << <countBlock, countThread >> > (devCount, devOffset, devIndexes, devResult);

		cudaMemcpy(&resultGpu, devResult, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(indexesGpu.data(), devIndexes, reserveCount * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(devCount);
		cudaFree(devOffset);
		cudaFree(devResult);
		cudaFree(devIndexes);
	}

	printf("\nTest::RunTestIndex END.\n");

	if (resultCpu == resultGpu) {
		printf("Test::RunTestIndex result [%i, %i] OK.\n", resultCpu, resultGpu);
	}
	else {
		printf("Test::RunTestIndex result [%i, %i] FAIL.\n", resultCpu, resultGpu);
	}

	std::sort(indexesCpu.begin(), indexesCpu.end());
	std::sort(indexesGpu.begin(), indexesGpu.end());

	for (size_t i = 0; i < reserveCount; ++i) {
		printf("\tindex: %i: [%i, %i]\n", i, indexesCpu[i], indexesGpu[i]);
	}
}

#else

	void CUDA_Test::Run() { }
	void CUDA_Test::RunTestIndex() { }

#endif
