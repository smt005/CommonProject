
#include "WrapperX1.h"

bool WrapperX1::sync = true;
int WrapperX1::tag = 0;
int WrapperX1::tagCurrent = -1;

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <iostream>
#include <string>
#include <thread>

#include "Emulate.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////
/// CPU
//////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
	void CalcForcesCpu(unsigned int* count, unsigned int* offset, cuda::Vector3* positions, float* masses, cuda::Vector3* forces, CUDA_TEST::Index indexData) {
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
			cuda::Vector3* pos = &positions[index];
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

	void UpdatePositionsCpu(unsigned int* count, unsigned int* offset, cuda::Vector3* positions, cuda::Vector3* velocities, float* masses, cuda::Vector3* forces, float* dt, CUDA_TEST::Index indexData) {
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

			positions[index].x += velocities[index].x * *dt;
			positions[index].y += velocities[index].y * *dt;
			positions[index].z += velocities[index].z * *dt;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void WrapperX1::CalculateForceCPU(cuda::Buffer& buffer) {
	const unsigned int maxCountBlock = 1;
	const unsigned int maxCountThread = CUDA::multithread ? std::thread::hardware_concurrency() : 1;

	unsigned int countBlock;
	unsigned int countThread;
	unsigned int offset;
	unsigned int count = buffer.count;

	CUDA::GetOffsets(count, maxCountBlock, maxCountThread, countBlock, countThread, offset);

	unsigned int* devCount = &count;
	unsigned int* devOffset = &offset;
	cuda::Vector3* devPositions = buffer.positions.data();
	float* devMasses = buffer.masses.data();
	cuda::Vector3* devForces = buffer.forces.data();

	if (countThread > 1) {
		CUDA_TEST::CUDA_Emulate(countBlock, countThread, [&](CUDA_TEST::Index indexData) {
			CalcForcesCpu(devCount, devOffset, devPositions, devMasses, devForces, indexData);
		});
	}
	else {
		CUDA_TEST::Index indexData(0, 0, 1);
		CalcForcesCpu(devCount, devOffset, devPositions, devMasses, devForces, indexData);
	}
}

void WrapperX1::UpdatePositionCPU(cuda::Buffer& buffer, float dt) {
	const unsigned int maxCountBlock = 1;
	const unsigned int maxCountThread = CUDA::multithread ? std::thread::hardware_concurrency() : 1;

	unsigned int countBlock;
	unsigned int countThread;
	unsigned int offset;
	unsigned int count = buffer.count;

	CUDA::GetOffsets(count, maxCountBlock, maxCountThread, countBlock, countThread, offset);

	unsigned int* devCount = &count;
	unsigned int* devOffset = &offset;
	cuda::Vector3* devPositions = buffer.positions.data();
	float* devMasses = buffer.masses.data();
	cuda::Vector3* devForces = buffer.forces.data();
	cuda::Vector3* devVelocities = buffer.velocities.data();
	float* devDt = &dt;

	if (countThread > 1) {
		CUDA_TEST::CUDA_Emulate(countBlock, countThread, [&](CUDA_TEST::Index indexData) {
			UpdatePositionsCpu(devCount, devOffset, devPositions, devVelocities, devMasses, devForces, devDt, indexData);
		});
	}
	else {
		CUDA_TEST::Index indexData(0, 0, 1);
		UpdatePositionsCpu(devCount, devOffset, devPositions, devVelocities, devMasses, devForces, devDt, indexData);
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
// GPU ///////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

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

				atomicAdd(&forces[index].x, gravityX);
				atomicAdd(&forces[index].y, gravityY);
				atomicAdd(&forces[index].z, gravityZ);
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

				atomicAdd(&forces[index].x, gravityX);
				atomicAdd(&forces[index].y, gravityY);
				atomicAdd(&forces[index].z, gravityZ);
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

			positions[index].x += velocities[index].x * *dt;
			positions[index].y += velocities[index].y * *dt;
			positions[index].z += velocities[index].z * *dt;
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

void WrapperX1::CalculateForceGPU(cuda::Buffer& buffer) {
	const unsigned int maxCountBlock = 65535;
	const unsigned int maxCountThread = CUDA::multithread ? CUDA::maxThreadsPerBlock : 1;

	unsigned int count = buffer.count;
	unsigned int countBlock;
	unsigned int countThread;
	unsigned int offset;

	CUDA::GetOffsets(count, maxCountBlock, maxCountThread, countBlock, countThread, offset);

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
	cudaMemcpy(devPositions, buffer.positions.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(devMasses, buffer.masses.data(), count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devForces, buffer.forces.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);

	if (sync) {
		CalcForcesGpuSync << <countBlock, countThread >> > (devCount, devOffset, devPositions, devMasses, devForces);
	}
	else {
		CalcForcesGpu << <countBlock, countThread >> > (devCount, devOffset, devPositions, devMasses, devForces);
	}

	cudaMemcpy(buffer.forces.data(), devForces, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);

	cudaFree(devCount);
	cudaFree(devOffset);
	cudaFree(devPositions);
	cudaFree(devMasses);
	cudaFree(devForces);
}

void WrapperX1::UpdatePositionGPU(cuda::Buffer& buffer, float dt) {
	const unsigned int maxCountBlock = 65535;
	const unsigned int maxCountThread = CUDA::multithread ? CUDA::maxThreadsPerBlock : 1;

	unsigned int count = buffer.count;
	unsigned int countBlock;
	unsigned int countThread;
	unsigned int offset;

	CUDA::GetOffsets(count, maxCountBlock, maxCountThread, countBlock, countThread, offset);

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
	cudaMemcpy(devPositions, buffer.positions.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(devMasses, buffer.masses.data(), count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devForces, buffer.forces.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(devVelocities, buffer.velocities.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);
	cudaMemcpy(devDt, &dt, sizeof(float), cudaMemcpyHostToDevice);

	UpdatePositionsGpu << <countBlock, countThread >> > (devCount, devOffset, devPositions, devVelocities, devMasses, devForces, devDt);

	cudaMemcpy(buffer.positions.data(), devPositions, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);

	cudaFree(devCount);
	cudaFree(devOffset);
	cudaFree(devPositions);
	cudaFree(devMasses);
	cudaFree(devForces);
	cudaFree(devVelocities);
	cudaFree(devDt);
}
