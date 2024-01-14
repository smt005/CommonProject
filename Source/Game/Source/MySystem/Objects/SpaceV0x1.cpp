#include "SpaceV0x1.h"
#include <thread>
#include <../../CUDA/Wrapper.h>
#include <../../CUDA/WrapperV0x1.h>

void SpaceV0x1::Update(double dt) {
	size_t sizeData = _datas.size();
	if (sizeData <= 1) {
		return;
	}

	unsigned int count = _bodies.size();
	CUDA::Vector3* positions = new CUDA::Vector3[count];
	float* masses = new float[count];
	CUDA::Vector3* forces = new CUDA::Vector3[count];
	CUDA::Vector3* velocities = new CUDA::Vector3[count];

	for (size_t index = 0; index < _bodies.size(); ++index) {
		Body& body = *_bodies[index];

		CUDA::Vector3& pos = positions[index];
		auto bodyPos = body.GetPos();
		pos.x = bodyPos.x;
		pos.y = bodyPos.y;
		pos.z = bodyPos.z;

		masses[index] = body._mass;

		CUDA::Vector3& velocity = velocities[index];
		velocity.x = body._velocity.x;
		velocity.y = body._velocity.y;
		velocity.z = body._velocity.z;
	}

	if (processGPU) {
		WrapperV0x1::UpdatePositionGPU(count, positions, masses, forces, velocities);
	} else {
		WrapperV0x1::UpdatePositionCPU(count, positions, masses, forces, velocities);
	}

	delete[] positions;
	delete[] masses;
	delete[] forces;
	delete[] velocities;
}
