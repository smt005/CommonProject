#include "SpaceGpuX0.h"
#include <algorithm>
#include <stdio.h>
#include <Core.h>
#include <../../CUDA/Wrapper.h>
#include <../../CUDA/WrapperX0.h>

void SpaceGpuX0::Update(double dt) {
	if (countOfIteration == 0) {
		return;
	}
	size_t sizeData = _bodies.size();
	if (sizeData <= 1) {
		return;
	}

	unsigned int count = _bodies.size();

	if (processGPU) {
		WrapperX0::UpdatePositionGPU(count, _positions.data(), _masses.data(), _forces.data(), _velocities.data(), deltaTime, countOfIteration);
	} else {
		WrapperX0::UpdatePositionCPU(count, _positions.data(), _masses.data(), _forces.data(), _velocities.data(), deltaTime, countOfIteration);
	}

	for (size_t index = 0; index < count; ++index) {
		_bodies[index]->SetPos(Math::Vector3d(_positions[index].x, _positions[index].y, _positions[index].z));
	}
}

void SpaceGpuX0::Preparation() {
	double lastTime = Engine::Core::currentTime();

	_positions.clear();
	_masses.clear();
	_forces.clear();
	_velocities.clear();

	size_t count = _bodies.size();
	if (count == 0) {

		return;
	}

	std::sort(_bodies.begin(), _bodies.end(), [](const Body::Ptr& left, const Body::Ptr& right) {
		if (left && right) {
			return left->_mass > right->_mass;
		}
		return left && !right;
	});

	_positions.reserve(count);
	_masses.reserve(count);
	_velocities.reserve(count);
	_forces.resize(count);

	for (Body::Ptr& body : _bodies) {
		body->Scale();

		auto pos = body->GetPos();
		_positions.emplace_back(CUDA::Vector3(pos.x, pos.y, pos.z));
		_masses.emplace_back(body->_mass);
		_velocities.emplace_back(body->_velocity.x, body->_velocity.y, body->_velocity.z);
	}

	size_t sizeInfo = 10;
	sizeInfo = sizeInfo > _bodies.size() ? _bodies.size() : 10;
	_heaviestInfo.clear();
	_heaviestInfo.reserve(sizeInfo);

	for (size_t index = 0; index < sizeInfo; ++index) {
		if (Body::Ptr& body = _bodies[index]) {
			_heaviestInfo.emplace_back(body, std::to_string(body->_mass));
		}
	}

	lastTime = Engine::Core::currentTime() - lastTime;
	printf("SpaceGpuX0::Preparation: %f size: %i\n", lastTime, _bodies.size());
}

std::string SpaceGpuX0::GetNameClass() {
	return Engine::GetClassName(this);
}
