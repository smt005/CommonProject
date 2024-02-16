#include "SpaceTree01.h"
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <unordered_map>>
#include <set>
#include <Core.h>
#include "BodyData.h"

using Cluster = spaceTree01::Cluster01;
//.............................................................
std::string SpaceTree01::GetNameClass() {
	return Engine::GetClassName(this);
}

void SpaceTree01::Update(double dt) {
	if (countOfIteration == 0 || _bodies.size() <= 1) {
		return;
	}

	for (size_t i = 0; i < countOfIteration; ++i) {
		Update();
	}
}

void SpaceTree01::Update() {
	if (buffer.empty()) {
		Preparation();
	}
}

void SpaceTree01::Preparation() {
	_debugInfo.Clear();

	size_t maxCountBodies = 100;

	mystd::shared<std::vector<Cluster::Ptr>> bufferPtr(new std::vector<Cluster::Ptr>());
	mystd::shared<std::vector<Cluster::Ptr>> newBufferPtr(new std::vector<Cluster::Ptr>());
	
	{
		Cluster& firstCluster = *bufferPtr->emplace_back(new Cluster());
		firstCluster.bodies.reserve(_bodies.size());

		Math::Vector3& min = firstCluster.min;
		Math::Vector3& max = firstCluster.max;
		min = Math::Vector3(std::numeric_limits<float>::max());
		max = Math::Vector3(std::numeric_limits<float>::min());

		for (Body::Ptr& body : _bodies) {
			Math::Vector3 pos = body->GetPos();

			min.x = pos.x < min.x ? pos.x : min.x;
			min.y = pos.y < min.y ? pos.y : min.y;
			min.z = pos.z < min.z ? pos.z : min.z;
			max.x = pos.x > max.x ? pos.x : max.x;
			max.y = pos.y > max.y ? pos.y : max.y;
			max.z = pos.z > max.z ? pos.z : max.z;

			firstCluster.bodies.emplace_back(body.get());
		}

		_debugInfo.min = min;
		_debugInfo.max = max;
		_debugInfo.size = max - min;
		_debugInfo.countBodies = _bodies.size();
	}

	while (true) {
		for (Cluster::Ptr& cluster : *bufferPtr) {
			size_t countBodies = cluster->bodies.size();

			if (countBodies <= maxCountBodies && countBodies > 0) {
				float& sumMass = cluster->mass = 0;
				Math::Vector3 sumMassPos;

				for (Body* body : cluster->bodies) {
					Math::Vector3 pos = body->GetPos();

					float mass = body->Mass();
					sumMass += mass;
					sumMassPos += (pos * mass);

					body->CalcScale();
				}

				sumMassPos / sumMass;
				buffer.emplace_back(cluster);

				{
					_debugInfo.minBodies = countBodies < _debugInfo.minBodies ? countBodies : _debugInfo.minBodies;
					_debugInfo.maxBodies = countBodies > _debugInfo.maxBodies ? countBodies : _debugInfo.maxBodies;

					Math::Vector3 size = cluster->max - cluster->min;
					float volume = size.x * size.y; // * size.z;

					float minVolume = _debugInfo.minSize.x * _debugInfo.minSize.y; // * minSize.z;
					if (volume < minVolume) {
						_debugInfo.minSize = size;
					}
					float maxVolume = _debugInfo.maxSize.x * _debugInfo.maxSize.y; // * maxSize.z;
					if (volume > maxVolume) {
						_debugInfo.maxSize = size;
					}
				}

				continue;
			}

			std::vector<Cluster::Ptr> tempBuffer;
			tempBuffer.reserve(4);

			//...
			Math::Vector3& min = cluster->min;
			Math::Vector3& max = cluster->max;
			Math::Vector3 half = min + (max - min) / 2.f;

			{
				Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
				cluster->min.x = min.x;
				cluster->max.x = half.x;
				cluster->min.y = min.y;
				cluster->max.y = half.y;
			}

			{
				Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
				cluster->min.x = min.x;
				cluster->max.x = half.x;
				cluster->min.y = half.y;
				cluster->max.y = max.y;
			}

			{
				Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
				cluster->min.x = half.x;
				cluster->max.x = max.x;
				cluster->min.y = min.y;
				cluster->max.y = half.y;
			}

			{
				Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
				cluster->min.x = half.x;
				cluster->max.x = max.x;
				cluster->min.y = half.y;
				cluster->max.y = max.y;
			}

			//...
			for (Body* body : cluster->bodies) {
				auto pos = body->GetPos();

				if (pos.x <= half.x && pos.y <= half.y) {
					tempBuffer[0]->bodies.emplace_back(body);
				}
				else
				if (pos.x <= half.x && pos.y > half.y) {
					tempBuffer[1]->bodies.emplace_back(body);
				}
				else
				if (pos.x > half.x && pos.y <= half.y) {
					tempBuffer[2]->bodies.emplace_back(body);
				}
				else
				if (pos.x > half.x && pos.y > half.y) {
					tempBuffer[3]->bodies.emplace_back(body);
				}
			}

			//...
			for (Cluster::Ptr& cluster : tempBuffer) {
				if (!cluster->bodies.empty()) {
					newBufferPtr->emplace_back(cluster);
				}
			}
		}

		if (newBufferPtr->empty()) {
			break;
		}

		std::swap(newBufferPtr, bufferPtr);
		newBufferPtr->clear();
		++_debugInfo.countLevel;
	}

	// DEBUG
	_debugInfo.countCluster = buffer.size();
	
	_debugInfo.Print();
}

void spaceTree01::DebugInfo::Clear() {
	countLevel = 0;
	countCluster = 0;
	countBodies = 0;
	minBodies = std::numeric_limits<float>::max();
	maxBodies = 0;

	min = Math::Vector3(std::numeric_limits<float>::max());
	max = Math::Vector3(0.f);
	size = Math::Vector3(0.f);
	minSize = Math::Vector3(std::numeric_limits<float>::max());
	maxSize = Math::Vector3(0.f);
}

void spaceTree01::DebugInfo::Print() {
	printf("Clusters info.\n");
	printf("count cluster: %i count body: %i \n", countCluster, countBodies);
	printf("count level: %i minBodies: %i maxBodies: %i \n", countLevel, minBodies, maxBodies);

	printf("\nSize [%f, %f, %f]\n", size.x, size.y, size.z);
	printf("Min size [%f, %f, %f]\n", minSize.x, minSize.y, minSize.z);
	printf("Max size [%f, %f, %f]\n", maxSize.x, maxSize.y, maxSize.z);
}
