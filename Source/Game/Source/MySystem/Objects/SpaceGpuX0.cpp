#include "SpaceGpuX0.h"
#include <algorithm>
#include <stdio.h>
#include <Core.h>
#include "../../CUDA/Source/Wrapper.h"
#include <../../CUDA/Source/WrapperX0.h>

void SpaceGpuX0::Update(double dt) {
	if (countOfIteration == 0) {
		return;
	}
	size_t sizeData = _bodies.size();
	if (sizeData <= 1) {
		return;
	}

	unsigned int count = _bodies.size();

	debugInfo.countOperation = count * count;

	if (processGPU) {
		WrapperX0::UpdatePositionGPU(count, _positions.data(), _masses.data(), _forces.data(), _velocities.data(), deltaTime, countOfIteration);
	} else {
		WrapperX0::UpdatePositionCPU(count, _positions.data(), _masses.data(), _forces.data(), _velocities.data(), deltaTime, countOfIteration);
	}

	for (size_t index = 0; index < count; ++index) {
		_bodies[index]->SetPos(Math::Vector3(_positions[index].x, _positions[index].y, _positions[index].z));
	}

	printf("OPERATIONS: %i\n", debugInfo.countOperation);
}

void SpaceGpuX0::Preparation() {
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
			return left->Mass() > right->Mass();
		}
		return left && !right;
	});

	_positions.reserve(count);
	_masses.reserve(count);
	_velocities.reserve(count);
	_forces.resize(count);

	for (Body::Ptr& body : _bodies) {
		body->CalcScale();

		auto pos = body->GetPos();
		_positions.emplace_back(CUDA::Vector3(pos.x, pos.y, pos.z));
		_masses.emplace_back(body->Mass());
		_velocities.emplace_back(body->Velocity().x, body->Velocity().y, body->Velocity().z);
	}

	size_t sizeInfo = 10;
	sizeInfo = sizeInfo > _bodies.size() ? _bodies.size() : 10;
	_heaviestInfo.clear();
	_heaviestInfo.reserve(sizeInfo);

	for (size_t index = 0; index < sizeInfo; ++index) {
		if (Body::Ptr& body = _bodies[index]) {
			_heaviestInfo.emplace_back(body, std::to_string(body->Mass()));
		}
	}
}

std::string SpaceGpuX0::GetNameClass() {
	return Engine::GetClassName(this);
}
