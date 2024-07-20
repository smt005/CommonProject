// ◦ Xyz ◦

#include "SpaceGpuX1.h"
#include <algorithm>
#include <stdio.h>
#include <unordered_map>>
#include <set>
#include <Core.h>
#include <../../CUDA/Source/Wrapper.h>
#include <../../CUDA/Source/WrapperX0.h>
#include <../../CUDA/Source/WrapperX1.h>
#include "BodyData.h"

void SpaceGpuX1::Update(double dt)
{
	if (countOfIteration == 0 || _bodies.size() <= 1) {
		return;
	}

	for (size_t i = 0; i < countOfIteration; ++i) {
		Update();
	}
}

void SpaceGpuX1::Update()
{
	unsigned int count = _bodies.size();

	if (tag == 0) {
		size_t countRemove = 0;

		auto collapsBodies = [&]() {
			if (buffer.countCollisions > 0) {
				std::unordered_map<int, std::set<int>*> collisionMap;
				std::vector<std::set<int>*> collisionVector;
				std::vector<cuda::Buffer::Pair>& collisions = buffer.collisions;

				for (unsigned int i = 0; i < buffer.countCollisions; ++i) {
					unsigned int firstIndex = collisions[i].first;
					unsigned int secondIndex = collisions[i].second;

					auto itFirst = collisionMap.find(firstIndex);
					std::set<int>* associateSet = nullptr;

					if (itFirst != collisionMap.end()) {
						associateSet = itFirst->second;
					}
					else {
						associateSet = new std::set<int>();
						associateSet->emplace(firstIndex);

						collisionVector.emplace_back(associateSet);
						collisionMap.emplace(firstIndex, associateSet);
					}

					collisionMap.emplace(secondIndex, associateSet);
					associateSet->emplace(secondIndex);
				}

				for (std::set<int>* setPtr : collisionVector) {
					auto& bodyIndexes = *setPtr;

					BodyData* bodyPtr = nullptr;

					float sumMass = 0;
					cuda::Vector3 sumForce;
					Math::Vector3 sumPulse;
					Math::Vector3 sumMassPos;
					Math::Vector3 velocity;
					unsigned int firstIndex = 0;

					for (unsigned int index : bodyIndexes) {
						sumMass += buffer.masses[index];

						sumMassPos.x += buffer.positions[index].x * buffer.masses[index];
						sumMassPos.y += buffer.positions[index].y * buffer.masses[index];
						sumMassPos.z += buffer.positions[index].z * buffer.masses[index];

						sumPulse.x += buffer.velocities[index].x * buffer.masses[index];
						sumPulse.y += buffer.velocities[index].y * buffer.masses[index];
						sumPulse.z += buffer.velocities[index].z * buffer.masses[index];

						sumForce.x += buffer.forces[index].x;
						sumForce.y += buffer.forces[index].y;
						sumForce.z += buffer.forces[index].z;

						if (!bodyPtr) {
							bodyPtr = static_cast<BodyData*>(_bodies[index].get());
							firstIndex = index;
						}
						else {
							_bodies[index]->visible = false;
							//_bodies[index].reset();

							buffer.masses[index] = -1.0;
							++countRemove;
						}
					}

					Math::Vector3 pos = sumMassPos / sumMass;
					sumPulse /= sumMass;

					bodyPtr->SetPos(pos);
					bodyPtr->_velocity = sumPulse;
					bodyPtr->_mass = sumMass;
					bodyPtr->CalcScale();

					buffer.masses[firstIndex] = sumMass;

					buffer.positions[firstIndex].x = pos.x;
					buffer.positions[firstIndex].y = pos.y;
					buffer.positions[firstIndex].z = pos.z;

					buffer.velocities[firstIndex].x = sumPulse.x;
					buffer.velocities[firstIndex].y = sumPulse.y;
					buffer.velocities[firstIndex].z = sumPulse.z;

					buffer.forces[firstIndex].x = sumForce.x;
					buffer.forces[firstIndex].y = sumForce.y;
					buffer.forces[firstIndex].z = sumForce.z;

					delete setPtr;
				}
			}
		};

		if (processGPU) {
			WrapperX1::CalculateForceGPU(buffer);
			collapsBodies();
			WrapperX1::UpdatePositionGPU(buffer, deltaTime);
		}
		else {
			WrapperX1::CalculateForceCPU(buffer);
			collapsBodies();
			WrapperX1::UpdatePositionCPU(buffer, deltaTime);
		}

		if (buffer.countCollisions == 0) {
			count = _bodies.size();
			for (size_t index = 0; index < count; ++index) {
				_bodies[index]->SetPos(Math::Vector3(buffer.positions[index].x, buffer.positions[index].y, buffer.positions[index].z));
			}

			buffer.Reset();
		}
		else {
			count = _bodies.size();
			std::vector<Body::Ptr> newBodies;
			newBodies.reserve(_bodies.size() - countRemove);
			size_t index = 0;

			for (auto bodyPtr : _bodies) {
				if (bodyPtr->visible) {
					bodyPtr->SetPos(Math::Vector3(buffer.positions[index].x, buffer.positions[index].y, buffer.positions[index].z));

					bodyPtr->Velocity().x = buffer.velocities[index].x;
					bodyPtr->Velocity().y = buffer.velocities[index].y;
					bodyPtr->Velocity().z = buffer.velocities[index].z;

					bodyPtr->Mass() = buffer.masses[index];

					newBodies.emplace_back(bodyPtr);
				}

				++index;
			}

			std::swap(_bodies, newBodies);
			buffer.Load<Body::Ptr>(_bodies);
		}
	}
	else {
		if (processGPU) {
			WrapperX0::UpdatePositionGPU(count, _positions.data(), _masses.data(), _forces.data(), _velocities.data(), deltaTime, countOfIteration);
		}
		else {
			WrapperX0::UpdatePositionCPU(count, _positions.data(), _masses.data(), _forces.data(), _velocities.data(), deltaTime, countOfIteration);
		}

		for (size_t index = 0; index < count; ++index) {
			_bodies[index]->SetPos(Math::Vector3(_positions[index].x, _positions[index].y, _positions[index].z));
		}
	}
}

void SpaceGpuX1::Preparation()
{
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

	buffer.Load<Body::Ptr>(_bodies);

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

std::string SpaceGpuX1::GetNameClass()
{
	return Engine::GetClassName(this);
}
