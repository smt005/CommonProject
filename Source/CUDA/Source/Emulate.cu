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
		unsigned int sumThreads = countBlock * countThread;
		if (sumThreads == 1) {
			fun(CUDA_TEST::Index(0, 0, 1));
			return;
		}

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
	
	double CUDA_Emulate::Time() {
		std::chrono::milliseconds ms;
		ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
		return static_cast<double>(ms.count());
	}

	unsigned int CUDA_Emulate::cudaMemcpyHostToDevice = 1;
	unsigned int CUDA_Emulate::cudaMemcpyDeviceToHost = 2;
}

/// CPU //////////////////////////////////////////////////////////////////////////////

namespace {
	void CalcForcesCpu(unsigned int* count, unsigned int* offset, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces, CUDA_TEST::Index indexData) {
		int indexT = indexData.threadIdx.x + indexData.blockIdx.x * indexData.blockDim.x;
		int startIndex = indexT * *offset;
		int countIndex = startIndex + *offset;
		if (countIndex >= *count) {
			countIndex = *count;
		}

		float constGravity = 0.01;
		float gravityX = 0.0;
		float gravityY = 0.0;
		float gravityZ = 0.0;
		float dist = 0.0;
		float force = 0.0;

		for (int index = startIndex; index < countIndex; ++index) {
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

	void UpdatePositionsCpu(unsigned int* count, unsigned int* offset, CUDA::Vector3* positions, CUDA::Vector3* velocities, float* masses, CUDA::Vector3* forces, float* dt, CUDA_TEST::Index indexData) {
		int indexT = indexData.threadIdx.x + indexData.blockIdx.x * indexData.blockDim.x;
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

/// GPU //////////////////////////////////////////////////////////////////////////////

namespace {
	__global__ void CalcForcesGpu(unsigned int* count, unsigned int* offset, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces) {
		int indexT = threadIdx.x + blockIdx.x * blockDim.x;
		int startIndex = indexT * *offset;
		int countIndex = startIndex + *offset;
		if (countIndex >= *count) {
			countIndex = *count;
		}

		float constGravity = 0.01;
		float gravityX = 0.0;
		float gravityY = 0.0;
		float gravityZ = 0.0;
		float dist = 0.0;
		float force = 0.0;

		for (int index = startIndex; index < countIndex; ++index) {
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

	__global__ void CalcForcesGpuSync(unsigned int* count, unsigned int* offset, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces) {
		int indexT = threadIdx.x + blockIdx.x * blockDim.x;
		int startIndex = indexT * *offset;
		int countIndex = startIndex + *offset;
		if (countIndex >= *count) {
			countIndex = *count;
		}

		float constGravity = 0.01;
		float gravityX = 0.0;
		float gravityY = 0.0;
		float gravityZ = 0.0;
		float dist = 0.0;
		float force = 0.0;

		for (int index = startIndex; index < countIndex; ++index) {
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

				//atomicAdd(&forces[index].x, gravityX);
				//atomicAdd(&forces[index].y, gravityY);
				//atomicAdd(&forces[index].z, gravityZ);
			}
		}
	}

	__global__ void UpdatePositionsGpu(unsigned int* count, unsigned int* offset, CUDA::Vector3* positions, CUDA::Vector3* velocities, float* masses, CUDA::Vector3* forces, float* dt) {
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

/////////////////////////////////////////////////////////////////////////////////

void CUDA_TEST::Test(unsigned int count, bool sync) {
	printf("\nCUDA_TEST::Test count: %i, sync: %i .....................................................................\n", count, (int)sync);
	bool sameThread = false;
	unsigned int bbb = 1;
	unsigned int ttt = 16;

	std::vector<float> masses;
	masses.reserve(count);
	std::vector<CUDA::Vector3> positionsCpu;
	positionsCpu.reserve(count);
	std::vector<CUDA::Vector3> positionsGpu;
	positionsGpu.reserve(count);
	std::vector<CUDA::Vector3> velocitiesBoth;
	velocitiesBoth.reserve(count);

	//for (int i = (count-1); i >= 0; --i) {
	for (int i = 0; i < count; ++i) {
		positionsCpu.emplace_back((float)i, (float)i, (float)i);
		positionsGpu.emplace_back((float)i, (float)i, (float)i);
		masses.emplace_back((float)i + 1.f);
		velocitiesBoth.emplace_back((float)i, (float)i, (float)i);
	}

	float dt = 1.f;

	// CPU
	{
		std::vector<CUDA::Vector3>& positions = positionsCpu;
		std::vector<CUDA::Vector3> forces;
		forces.resize(count);
		std::vector<CUDA::Vector3> velocities = velocitiesBoth;

		const unsigned int maxCountBlock = 1;// sameThread ? bbb : 1;
		const unsigned int maxCountThread = 16;// sameThread ? ttt : 16;

		unsigned int countBlock;
		unsigned int countThread;
		unsigned int offset;

		CUDA::GetOffsets(count, maxCountBlock, maxCountThread, countBlock, countThread, offset);
		printf("\nCPU BEGIN count: %i, offset %i, countBlock %i, countThread %i\n", count, offset, countBlock, countThread);
		double timeAll = CUDA_TEST::CUDA_Emulate::Time();

		unsigned int* devCount;
		unsigned int* devOffset;
		CUDA::Vector3* devPositions;
		float* devMasses;
		CUDA::Vector3* devForces;
		CUDA::Vector3* devVelocities;
		float* devDt;

		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devCount, sizeof(unsigned int));
		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devOffset, sizeof(unsigned int));
		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devPositions, count * sizeof(CUDA::Vector3));
		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devMasses, count * sizeof(float));
		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devForces, count * sizeof(CUDA::Vector3));
		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devVelocities, count * sizeof(CUDA::Vector3));
		CUDA_TEST::CUDA_Emulate::cudaMalloc(&devDt, sizeof(float));

		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devCount, &count, sizeof(unsigned int), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devOffset, &offset, sizeof(unsigned int), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devPositions, positions.data(), count * sizeof(CUDA::Vector3), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devMasses, masses.data(), count * sizeof(float), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devForces, forces.data(), count * sizeof(CUDA::Vector3), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devVelocities, velocities.data(), count * sizeof(CUDA::Vector3), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(devDt, &dt, sizeof(float), CUDA_TEST::CUDA_Emulate::cudaMemcpyHostToDevice);

		double time = CUDA_TEST::CUDA_Emulate::Time();

		CUDA_Emulate(countBlock, countThread, [&](Index indexData) {
			CalcForcesCpu(devCount, devOffset, devPositions, devMasses, devForces, indexData);
		});

		CUDA_TEST::CUDA_Emulate::cudaMemcpy(forces.data(), devForces, count * sizeof(CUDA::Vector3), CUDA_TEST::CUDA_Emulate::cudaMemcpyDeviceToHost);

		CUDA_Emulate(countBlock, countThread, [&](Index indexData) {
			UpdatePositionsCpu(devCount, devOffset, devPositions, devVelocities, devMasses, devForces, devDt, indexData);
		});

		CUDA_TEST::CUDA_Emulate::cudaMemcpy(positions.data(), devPositions, count * sizeof(CUDA::Vector3), CUDA_TEST::CUDA_Emulate::cudaMemcpyDeviceToHost);
		CUDA_TEST::CUDA_Emulate::cudaMemcpy(velocities.data(), devVelocities, count * sizeof(CUDA::Vector3), CUDA_TEST::CUDA_Emulate::cudaMemcpyDeviceToHost);

		time = CUDA_TEST::CUDA_Emulate::Time() - time;

		CUDA_TEST::CUDA_Emulate::cudaFree(devCount);
		CUDA_TEST::CUDA_Emulate::cudaFree(devOffset);
		CUDA_TEST::CUDA_Emulate::cudaFree(devPositions);
		CUDA_TEST::CUDA_Emulate::cudaFree(devMasses);
		CUDA_TEST::CUDA_Emulate::cudaFree(devForces);
		CUDA_TEST::CUDA_Emulate::cudaFree(devVelocities);
		CUDA_TEST::CUDA_Emulate::cudaFree(devDt);

		timeAll = CUDA_TEST::CUDA_Emulate::Time() - timeAll;

		printf("\nCPU END time: %f timeAll: %f\n", time, timeAll);
	}

	// GPU
	{
		std::vector<CUDA::Vector3>& positions = positionsGpu;
		std::vector<CUDA::Vector3> forces;
		forces.resize(count);
		std::vector<CUDA::Vector3> velocities = velocitiesBoth;

		const unsigned int maxCountBlock = sameThread ? bbb : 65535;
		const unsigned int maxCountThread = sameThread ? ttt : 1024;

		unsigned int countBlock;
		unsigned int countThread;
		unsigned int offset;

		CUDA::GetOffsets(count, maxCountBlock, maxCountThread, countBlock, countThread, offset);
		printf("\nGPU BEGIN count: %i, offset %i, countBlock %i, countThread %i\n", count, offset, countBlock, countThread);
		double timeAll = CUDA_TEST::CUDA_Emulate::Time();

		unsigned int* devCount;
		unsigned int* devOffset;
		CUDA::Vector3* devPositions;
		float* devMasses;
		CUDA::Vector3* devForces;
		CUDA::Vector3* devVelocities;
		float* devDt;

		cudaMalloc(&devCount, sizeof(unsigned int));
		cudaMalloc(&devOffset, sizeof(unsigned int));
		cudaMalloc(&devPositions, count * sizeof(CUDA::Vector3));
		cudaMalloc(&devMasses, count * sizeof(float));
		cudaMalloc(&devForces, count * sizeof(CUDA::Vector3));
		cudaMalloc(&devVelocities, count * sizeof(CUDA::Vector3));
		cudaMalloc(&devDt, sizeof(float));

		cudaMemcpy(devCount, &count, sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(devOffset, &offset, sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(devPositions, positions.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);
		cudaMemcpy(devMasses, masses.data(), count * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(devForces, forces.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);
		cudaMemcpy(devVelocities, velocities.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);
		cudaMemcpy(devDt, &dt, sizeof(float), cudaMemcpyHostToDevice);

		double time = CUDA_TEST::CUDA_Emulate::Time();

		if (sync) {
			CalcForcesGpuSync << <countBlock, countThread >> > (devCount, devOffset, devPositions, devMasses, devForces);
		} else {
			CalcForcesGpu << <countBlock, countThread >> > (devCount, devOffset, devPositions, devMasses, devForces);
		}

		cudaMemcpy(forces.data(), devForces, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);

		UpdatePositionsGpu << <countBlock, countThread >> > (devCount, devOffset, devPositions, devVelocities, devMasses, devForces, devDt);

		cudaMemcpy(positions.data(), devPositions, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);
		cudaMemcpy(velocities.data(), devVelocities, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);

		time = CUDA_TEST::CUDA_Emulate::Time() - time;

		cudaFree(devCount);
		cudaFree(devOffset);
		cudaFree(devPositions);
		cudaFree(devMasses);
		cudaFree(devForces);
		cudaFree(devVelocities);
		cudaFree(devDt);

		timeAll = CUDA_TEST::CUDA_Emulate::Time() - timeAll;
		printf("\nGPU END time: %f timeAll: %f\n", time, timeAll);
	}

	bool equal = true;
	bool printData = false;
	{
		for (size_t i = 0; i < count; ++i) {
			if (i < 10 || i > (count - 10)) {
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
	}

	if (equal) {
		printf("\nCOMPARE OK \n...............................................\n");
	} else {
		printf("\nCOMPARE FAIL \n...............................................\n");
	}	
}

#else

void CUDA_TEST::Test(unsigned int count, bool sync) { }

#endif
