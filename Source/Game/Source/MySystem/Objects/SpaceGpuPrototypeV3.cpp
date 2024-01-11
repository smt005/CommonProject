#include "SpaceGpuPrototypeV3.h"
#include <thread>
#include <../../CUDA/WrapperPrototypeV3.h>

void SpaceGpuPrototypeV3::Update(double dt) {
	size_t sizeData = _datas.size();
	if (sizeData <= 1) {
		return;
	}

	int count = _datas.size();
	float* masses = new float[count];
	float* positionsX = new float[count];
	float* positionsY = new float[count];
	float* forcesX = new float[count];
	float* forcesY = new float[count];

	for (size_t index = 0; index < count; ++index) {
		Body::Data& data = _datas[index];

		masses[index] = data.mass;

		positionsX[index] = data.pos.x;
		positionsY[index] = data.pos.y;

		forcesX[index] = 0.f;
		forcesY[index] = 0.f;
	}

	if (processGPU) {
		CUDA_PrototypeV3::GetForcesGPUStatic(count, masses, positionsX, positionsY, forcesX, forcesY);
	} else {		
		CUDA_PrototypeV3::GetForcesCPUStatic(count, masses, positionsX, positionsY, forcesX, forcesY);
	}

	// ...
	for (size_t index = 0; index < sizeData; ++index) {
		Body::Ptr& body = _bodies[index];
		if (!body) {
			continue;
		}

		Math::Vector3d acceleration(forcesX[index] / body->_mass, forcesY[index] / body->_mass, 0.0);
		Math::Vector3d newVelocity = acceleration * static_cast<double>(dt);

		body->_velocity += newVelocity;

		body->_dataPtr->pos += body->_velocity * static_cast<double>(dt);
		body->SetPos(body->_dataPtr->pos);

		// Info
		body->force = body->_dataPtr->force.length();
	}

	delete[] masses;
	delete[] positionsX;
	delete[] positionsY;
	delete[] forcesX;
	delete[] forcesY;

	//...
	if (dt > 0) {
		++time;
	}
	else {
		--time;
	}
}