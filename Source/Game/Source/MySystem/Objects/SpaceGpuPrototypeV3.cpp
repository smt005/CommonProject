#include "SpaceGpuPrototypeV3.h"
#include <thread>
#include <../../CUDA/Wrapper.h>
#include <../../CUDA/WrapperPrototypeV3.h>

void SpaceGpuPrototypeV3::Update(double dt) {
	size_t sizeData = _datas.size();
	if (sizeData <= 1) {
		return;
	}

	int count = _bodies.size();
	std::vector<CUDA::Body> bodies;
	bodies.reserve(count);

	for (Body::Ptr& bodyPtr : _bodies) {
		auto pos = bodyPtr->GetPos();
		auto& vel = bodyPtr->_velocity;
		bodies.emplace_back(pos.x, pos.y, pos.z, bodyPtr->_mass, (float)vel.x, (float)vel.y, (float)vel.z);
	}

	std::vector<CUDA::Vector3> forces;

	if (processGPU) {
		CUDA_PrototypeV3::GetForcesGPUStatic(bodies, forces);
	} else {		
		CUDA_PrototypeV3::GetForcesCPUStatic(bodies, forces);
	}

	// ...
	for (size_t index = 0; index < sizeData; ++index) {
		Body::Ptr& body = _bodies[index];
		if (!body) {
			continue;
		}

		Math::Vector3d acceleration(forces[index].x / body->_mass, forces[index].y / body->_mass, forces[index].z / body->_mass);
		Math::Vector3d newVelocity = acceleration * static_cast<double>(dt);

		body->_velocity += newVelocity;

		body->_dataPtr->pos += body->_velocity * static_cast<double>(dt);
		body->SetPos(body->_dataPtr->pos);

		// Info
		body->force = body->_dataPtr->force.length();
	}

	//...
	if (dt > 0) {
		++time;
	}
	else {
		--time;
	}
}
