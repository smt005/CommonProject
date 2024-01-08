#include "SpaceGpuPrototype.h"
#include <thread>
#include <../CUDA/WrapperPrototype.h>

void SpaceGpuPrototype::Update(double dt) {
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
		CUDA_Prototype::GetForcesGPUStatic(count, masses, positionsX, positionsY, forcesX, forcesY);
	} else {		
		CUDA_Prototype::GetForcesCPUStatic(count, masses, positionsX, positionsY, forcesX, forcesY);
	}

	// ...
	float longÂistanceFromStar = 150000.f;
	size_t needDataAssociation = std::numeric_limits<double>::min();
	std::vector<size_t> indRem;

	size_t size = _bodies.size();

	Body::Ptr star = GetHeaviestBody();
	Math::Vector3d posStar = star ? star->GetPos() : Math::Vector3d();

	for (size_t index = 0; index < size; ++index) {
		Body::Ptr& body = _bodies[index];
		if (!body) {
			continue;
		}

		static double minForce = std::numeric_limits<double>::min();
		if ((body->_dataPtr->force.length() < minForce) && (star && (posStar - body->GetPos()).length() > longÂistanceFromStar)) {
			indRem.emplace_back(index);
			++needDataAssociation;
			continue;
		}

		Math::Vector3d acceleration = body->_dataPtr->force / body->_mass;
		Math::Vector3d newVelocity = acceleration * static_cast<double>(dt);

		body->_velocity += newVelocity;

		body->_dataPtr->pos += body->_velocity * static_cast<double>(dt);
		body->SetPos(body->_dataPtr->pos);

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