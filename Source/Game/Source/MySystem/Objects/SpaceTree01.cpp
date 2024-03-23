// ◦ Xyz ◦
#include "SpaceTree01.h"
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <unordered_map>>
#include <set>
#include <Core.h>
#include "BodyData.h"

using Cluster = spaceTree01::Cluster01;

std::string SpaceTree01::GetNameClass() {
	return Engine::GetClassName(this);
}

void SpaceTree01::LoadProperty() {
	std::string maxCountBodiesStr = _params["COUNT_IN_CLUSTER"];
	if (!maxCountBodiesStr.empty()) {
		maxCountBodies = std::stoi(maxCountBodiesStr);
	}

	std::string distFactorStr = _params["DIST_FACTOR"];
	if (!distFactorStr.empty()) {
		distFactor = std::stoi(distFactorStr);
	}

	std::string debugInfoStr = _params["DEBUG_INFO"];
	if (!debugInfoStr.empty()) {
		showDebugInfo = debugInfoStr == "true";
	}
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
	debugInfo.countOperation = 0;

	GenerateClusters();

	{
		double time = Engine::Core::currentTime();

		for (Cluster::Ptr& clusterPtr : buffer) {
			UpdateForceByBody(clusterPtr->bodies, clusterPtr->bodies);
			UpdateForceByCluster(*clusterPtr.get());
		}
		
		if (showDebugInfo) {
			double time1 = Engine::Core::currentTime();
			double dTime = time1 - time;
			printf("Update force: %lf\n", dTime);
		}
	}

	{
		double time = Engine::Core::currentTime();

		UpdatePos();

		if (showDebugInfo) {
			double time1 = Engine::Core::currentTime();
			double dTime = time1 - time;
			printf("Update pos: %lf\n", dTime);
		}
	}

	if (showDebugInfo) {
		printf("OPERATIONS: %i\n", debugInfo.countOperation);
	}
}

void SpaceTree01::UpdateForceByBody(std::vector<Body*>& bodies, std::vector<Body*>& subBodies) {
	for (Body* body : bodies) {
		auto pos = body->GetPos();
		float mass = body->Mass();
		Math::Vector3& forceVec = body->Force();

		for (Body* subBody : subBodies) {
			if (body == subBody) {
				continue;
			}

			Math::Vector3 gravityVec = subBody->GetPos() - pos;
			double dist = Math::length(gravityVec);
			if (dist == 0.f) {
				continue;
			}

			gravityVec = Math::normalize(gravityVec);
			double force = _constGravity * (mass * subBody->Mass()) / (dist * dist);
			gravityVec *= force;
			forceVec += gravityVec;

			++debugInfo.countOperation;
		}
	}
}

void SpaceTree01::UpdateForceByCluster(Cluster& cluster) {
	for (Body* body : cluster.bodies) {
		auto pos = body->GetPos();
		float mass = body->Mass();
		Math::Vector3& forceVec = body->Force();

		for (Cluster::Ptr& subClusterPtr : buffer) {
			if (&cluster == subClusterPtr.get()) {
				continue;
			}

			if (IsClosestCluster(cluster, *subClusterPtr)) {
				std::vector<Body*> bodiesTemp = { body };
				UpdateForceByBody(bodiesTemp, subClusterPtr->bodies);
			} else {
				Math::Vector3 gravityVec = subClusterPtr->centerMass - pos;
				double dist = Math::length(gravityVec);

				gravityVec = Math::normalize(gravityVec);
				double force = _constGravity * (mass * subClusterPtr->mass) / (dist * dist);
				gravityVec *= force;
				forceVec += gravityVec;

				++debugInfo.countOperation;
			}
		}
	}
}

bool SpaceTree01::IsClosestCluster(Cluster& cluster0, Cluster& cluster1) {
	float dist = (cluster0.centerPos - cluster1.centerPos).length();
	float closestDist = cluster0.dist + cluster1.dist;

	closestDist *= distFactor;
	return dist < closestDist;
}

void SpaceTree01::UpdatePos() {
	size_t size = _bodies.size();

	for (Body::Ptr& bodyPtr : _bodies) {
		Math::Vector3 acceleration = bodyPtr->Force() / bodyPtr->Mass();
		Math::Vector3 newVelocity = acceleration * deltaTime;

		bodyPtr->Velocity() += newVelocity;

		Math::Vector3 pos = bodyPtr->GetPos();
		pos += bodyPtr->Velocity() * deltaTime;
		bodyPtr->SetPos(pos);

		bodyPtr->Force() = 0.f;
	}
}

void SpaceTree01::Preparation() {
	buffer.clear();
}

void SpaceTree01::GenerateClusters() {
	double time = Engine::Core::currentTime();
	//_debugInfo.Clear();

	buffer.clear();

	mystd::shared<std::vector<Cluster::Ptr>> bufferPtr(new std::vector<Cluster::Ptr>());
	mystd::shared<std::vector<Cluster::Ptr>> newBufferPtr(new std::vector<Cluster::Ptr>());
	
	{
		Cluster& firstCluster = *bufferPtr->emplace_back(new Cluster());
		firstCluster.bodies.reserve(_bodies.size());

		Math::Vector3 min = Math::Vector3(std::numeric_limits<float>::max());
		Math::Vector3 max = Math::Vector3(std::numeric_limits<float>::min());

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

		Math::Vector3 size = max - min;

		if (size.x > size.y) {
			float k = size.x / size.y;
			min.y *= k;
			max.y *= k;
		}
		else if (size.y > size.x) {
			float k = size.y / size.x;
			min.x *= k;
			max.x *= k;

		}

		size = max - min;

		firstCluster.min = min;
		firstCluster.max = max;

		_debugInfo.min = min;
		_debugInfo.max = max;
		_debugInfo.size = max - min;
		_debugInfo.countBodies = _bodies.size();
	}

	while (true) {
		for (Cluster::Ptr& cluster : *bufferPtr) {
			size_t countBodies = cluster->bodies.size();

			// Добавление
			if (countBodies <= maxCountBodies && countBodies > 0) {
				float sumMass = 0;
				Math::Vector3 sumMassPos;

				for (Body* body : cluster->bodies) {
					Math::Vector3 pos = body->GetPos();
					float mass = body->Mass();
					sumMass += mass;
					sumMassPos += (pos * mass);

					body->CalcScale();
				}

				cluster->centerMass = sumMassPos / sumMass;
				cluster->mass = sumMass;
				buffer.emplace_back(cluster);

				{
					Math::Vector3 size = cluster->max - cluster->min;
					Math::Vector3 sizeHalt = size / 2.f;
					cluster->dist = sizeHalt.length();
					cluster->centerPos = cluster->min + sizeHalt;

				}

				if (showDebugInfo) {
					_debugInfo.minBodies = countBodies < _debugInfo.minBodies ? countBodies : _debugInfo.minBodies;
					_debugInfo.maxBodies = countBodies > _debugInfo.maxBodies ? countBodies : _debugInfo.maxBodies;

					Math::Vector3 size = cluster->max - cluster->min;
					cluster->dist = size.length();
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
			//...

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

	if (showDebugInfo) {
		double time1 = Engine::Core::currentTime();
		double dTime = time1 - time;
		printf("Preparation:: %lf\n", dTime);

		{
			size_t countBodioes = 0;

			for (auto& b : buffer) {
				countBodioes += b->bodies.size();
			}

			printf("countBodioes:: %i\n", countBodioes);
		}
	}

	// DEBUG
	//_debugInfo.countCluster = buffer.size();
	//_debugInfo.Print();
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
