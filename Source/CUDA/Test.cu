#include "Test.h"

#if ENABLE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <string>
#include <thread>
#include <stdio.h>

#include "Wrapper.h"

namespace {
	void CalcForcesCpu(int* count, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces) {
		float constGravity = 0.01;
		int statIndex = 0;
		int countIndex = *count;

		float gravityX;
		float gravityY;
		float gravityZ;
		float dist = 0;
		float force = 0;

		// INFO
		printf("CalcForcesCpu statIndex: %i countIndex: %i\n", statIndex, countIndex);
		for (int index = statIndex; index < countIndex; ++index) {
			printf("CalcForcesCpu positions: [%f, %f, %f], masses: %f, forces: [%f, %f, %f]\n",
				positions[index].x, positions[index].y, positions[index].z,
				masses[index],
				forces[index].x, forces[index].y, forces[index].z);
		}

		for (int index = statIndex; index < countIndex; ++index) {
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

	void UpdatePositionsCpu(int* count, CUDA::Vector3* positions, CUDA::Vector3* velocities, float* masses, CUDA::Vector3* forces, float* dt) {
		int statIndex = 0;
		int countIndex = *count;

		float accelerationX;
		float accelerationY;
		float accelerationZ;
		float appendVelocityX;
		float appendVelocityY;
		float appendVelocityZ;

		// INFO
		printf("UpdatePositionsCpu statIndex: %i countIndex: %i, dt: %f\n", statIndex, countIndex, *dt);
		for (int index = statIndex; index < countIndex; ++index) {
			printf("UpdatePositionsCpu positions: [%f, %f, %f], velocities: [%f, %f, %f], masses: %f, forces: [%f, %f, %f]\n",
				positions[index].x, positions[index].y, positions[index].z,
				velocities[index].x, velocities[index].y, velocities[index].z,
				masses[index],
				forces[index].x, forces[index].y, forces[index].z);
		}

		for (int index = statIndex; index < countIndex; ++index) {
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

	__global__ void CalcForcesGpu(int* count, CUDA::Vector3* positions, float* masses, CUDA::Vector3* forces) {
		float constGravity = 0.01;
		int statIndex = 0;
		int countIndex = *count;

		float gravityX;
		float gravityY;
		float gravityZ;
		float dist = 0;
		float force = 0;

		// INFO
		// INFO
		printf("CalcForcesGpu statIndex: %i countIndex: %i\n", statIndex, countIndex);
		for (int index = statIndex; index < countIndex; ++index) {
			printf("CalcForcesCpu positions: [%f, %f, %f], masses: %f, forces: [%f, %f, %f]\n",
				positions[index].x, positions[index].y, positions[index].z,
				masses[index],
				forces[index].x, forces[index].y, forces[index].z);
		}

		for (int index = statIndex; index < countIndex; ++index) {
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

	__global__ void UpdatePositionsGpu(int* count, CUDA::Vector3* positions, CUDA::Vector3* velocities, float* masses, CUDA::Vector3* forces, float* dt) {
		int statIndex = 0;
		int countIndex = *count;

		float accelerationX;
		float accelerationY;
		float accelerationZ;
		float appendVelocityX;
		float appendVelocityY;
		float appendVelocityZ;

		// INFO
		printf("UpdatePositionsGpu statIndex: %i countIndex: %i, dt: %f\n", statIndex, countIndex, *dt);
		for (int index = statIndex; index < countIndex; ++index) {
			printf("UpdatePositionsCpu positions: [%f, %f, %f], velocities: [%f, %f, %f], masses: %f, forces: [%f, %f, %f]\n",
				positions[index].x, positions[index].y, positions[index].z,
				velocities[index].x, velocities[index].y, velocities[index].z,
				masses[index],
				forces[index].x, forces[index].y, forces[index].z);
		}

		for (int index = statIndex; index < countIndex; ++index) {
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

void CUDA_Test::Run() {
	int count = 10;

	printf("Test::Run begin count: %i.\n", count);
	
	static std::vector<CUDA::Vector3> positions;
	std::vector<float> masses;
	std::vector<CUDA::Vector3> velocities;
	float dt = 1.f;

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
		printf("Test::Run CPU begin\n");
		cpuPositions = positions;
		std::vector<CUDA::Vector3> cpuVelocities = velocities;

		std::vector<CUDA::Vector3> forces;
		forces.resize(count, CUDA::Vector3());

		CalcForcesCpu(&count, cpuPositions.data(), masses.data(), forces.data());
		UpdatePositionsCpu(&count, cpuPositions.data(), cpuVelocities.data(), masses.data(), forces.data(), &dt);

		printf("Test::Run CPU end\n\n");
	}

	// GPU
	std::vector<CUDA::Vector3> gpuPositions;
	{
		printf("Test::Run GPU begin\n");

		gpuPositions.resize(count);
		std::vector<CUDA::Vector3> gpuVelocities = velocities;

		int* devCount;
		float* devDt;
		CUDA::Vector3* devPositions;
		float* devMasses;
		CUDA::Vector3* devVelocities;
		CUDA::Vector3* devForces;

		cudaMalloc(&devCount,	sizeof(int));
		cudaMalloc(&devDt,	sizeof(float));
		cudaMalloc(&devPositions,	count * sizeof(CUDA::Vector3));
		cudaMalloc(&devMasses,	count * sizeof(float));
		cudaMalloc(&devVelocities,	count * sizeof(CUDA::Vector3));
		cudaMalloc(&devForces, count * sizeof(CUDA::Vector3));

		cudaMemcpy(devCount,	&count, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(devDt,	&dt, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(devPositions, positions.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);
		cudaMemcpy(devMasses, masses.data(), count * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(devVelocities, gpuVelocities.data(), count * sizeof(CUDA::Vector3), cudaMemcpyHostToDevice);

		CalcForcesGpu << <1, 1 >> > (devCount, devPositions, devMasses, devForces);
		UpdatePositionsGpu << <1, 1 >> > (devCount, devPositions, devVelocities, devMasses, devForces, devDt);

		cudaMemcpy(gpuPositions.data(), devPositions, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);
		cudaMemcpy(gpuVelocities.data(), devVelocities, count * sizeof(CUDA::Vector3), cudaMemcpyDeviceToHost);

		cudaFree(devCount);
		cudaFree(devDt);
		cudaFree(devPositions);
		cudaFree(devMasses);
		cudaFree(devForces);
		cudaFree(devVelocities);

		printf("Test::Run GPU end\n\n");
	}

	printf("Test::Run end.\n\n");

	//...
	bool equal = true;
	for (size_t i = 0; i < count; ++i) {
		if (!(cpuPositions[i].x == gpuPositions[i].x, cpuPositions[i].y == gpuPositions[i].y, cpuPositions[i].z == gpuPositions[i].z)) {
			equal = false;
			break;
		}
	}
	
	if (equal) {
		printf("Test::Run result OK.\n");
	}
	else {
		printf("Test::Run result FAIL.\n\n");

		for (size_t i = 0; i < count; ++i) {
			printf("\tpos: [%f, %f, %f] != [%f, %f, %f]\n", cpuPositions[i].x, cpuPositions[i].y, cpuPositions[i].z, gpuPositions[i].x, gpuPositions[i].y, gpuPositions[i].z);
		}
	}
}

#else

	void CUDA_Test::Run() { }

#endif
