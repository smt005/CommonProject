#include "SpaceTree02.h"
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <unordered_map>>
#include <set>
#include <Core.h>
#include "BodyData.h"

using Cluster = spaceTree02::Cluster02;

std::string SpaceTree02::GetNameClass() {
	return Engine::GetClassName(this);
}

void SpaceTree02::LoadProperty() {
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

void SpaceTree02::Update(double dt) {
	if (countOfIteration == 0 || _bodies.size() <= 1) {
		return;
	}

	for (size_t i = 0; i < countOfIteration; ++i) {
		Update();
	}
}

void SpaceTree02::Update() {
	debugInfo.countOperation = 0;

	GenerateClusters();

	for (Cluster::Ptr& clusterPtr : buffer) {
		UpdateForceByBody(clusterPtr->bodies, clusterPtr->bodies);
		UpdateForceByCluster(*clusterPtr.get());
	}

	UpdatePos();

	if (searchOptimalCountBodies) {
		printf("OPERATIONS: %i maxCount: %i\n", debugInfo.countOperation, maxCountBodies);

		if (debugInfo.countOperation < minCountOperation) {
			minCountOperation = debugInfo.countOperation;
			optimalCountBodies = maxCountBodies;
		}

		maxCountBodies += deltaOptimalCountBodies;

		if (maxCountBodies > maxOptimalCountBodies) {
			searchOptimalCountBodies = false;
			maxCountBodies = optimalCountBodies;

			printf("Optimal count bodies in cluster is: %i\n", optimalCountBodies);

			minCountOperation = std::numeric_limits<size_t>::max();
		}
	}
}

void SpaceTree02::UpdateForceByBody(std::vector<Body*>& bodies, std::vector<Body*>& subBodies) {
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

void SpaceTree02::UpdateForceByCluster(Cluster& cluster) {
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

bool SpaceTree02::IsClosestCluster(Cluster& cluster0, Cluster& cluster1) {
	float dist = (cluster0.centerPos - cluster1.centerPos).length();
	float closestDist = cluster0.dist + cluster1.dist;

	closestDist *= distFactor;
	return dist < closestDist;
}

void SpaceTree02::UpdatePos() {
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

void SpaceTree02::Preparation() {
	buffer.clear();
}

void SpaceTree02::GenerateClusters() {
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
		Math::Vector3 center = min + size / 2.f;

		float maxValue = 0;
		maxValue = size.x > maxValue ? size.x : maxValue;
		maxValue = size.y > maxValue ? size.y : maxValue;
		maxValue = size.z > maxValue ? size.z : maxValue;

		firstCluster.min.x = center.x - maxValue;
		firstCluster.min.y = center.y - maxValue;
		firstCluster.min.z = center.z - maxValue;
		firstCluster.max.x = center.x + maxValue;
		firstCluster.max.y = center.y + maxValue;
		firstCluster.max.z = center.z + maxValue;

		firstCluster.size = firstCluster.max - firstCluster.min;

		_debugInfo.min = firstCluster.min;
		_debugInfo.max = firstCluster.max;
		_debugInfo.size = firstCluster.max - firstCluster.min;
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

				cluster->size = cluster->max - cluster->min;

				buffer.emplace_back(cluster);

				{
					Math::Vector3 size = cluster->max - cluster->min;
					Math::Vector3 sizeHalt = size / 2.f;
					cluster->dist = sizeHalt.length();
					cluster->centerPos = cluster->min + sizeHalt;

				}

				continue;
			}
			//...

			Math::Vector3& min = cluster->min;
			Math::Vector3& max = cluster->max;
			Math::Vector3 half = min + (max - min) / 2.f;

			std::vector<Cluster::Ptr> tempBuffer = CreateSubClusters(min, max, half);

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
}

std::vector<Cluster::Ptr> SpaceTree02::CreateSubClusters(const Math::Vector3& min, const Math::Vector3& max, const Math::Vector3& half) {
	std::vector<Cluster::Ptr> tempBuffer;

	tempBuffer.reserve(8);

	// TOP
	{
		Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
		cluster->min.x = min.x;
		cluster->max.x = half.x;
		cluster->min.y = min.y;
		cluster->max.y = half.y;
		cluster->max.z = max.z;
		cluster->min.z = half.z;
	}

	{
		Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
		cluster->min.x = min.x;
		cluster->max.x = half.x;
		cluster->min.y = half.y;
		cluster->max.y = max.y;
		cluster->max.z = max.z;
		cluster->min.z = half.z;
	}

	{
		Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
		cluster->min.x = half.x;
		cluster->max.x = max.x;
		cluster->min.y = min.y;
		cluster->max.y = half.y;
		cluster->max.z = max.z;
		cluster->min.z = half.z;
	}

	{
		Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
		cluster->min.x = half.x;
		cluster->max.x = max.x;
		cluster->min.y = half.y;
		cluster->max.y = max.y;
		cluster->max.z = max.z;
		cluster->min.z = half.z;
	}

	// BOTTOM
	{
		Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
		cluster->min.x = min.x;
		cluster->max.x = half.x;
		cluster->min.y = min.y;
		cluster->max.y = half.y;
		cluster->max.z = half.z;
		cluster->min.z = min.z;
	}

	{
		Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
		cluster->min.x = min.x;
		cluster->max.x = half.x;
		cluster->min.y = half.y;
		cluster->max.y = max.y;
		cluster->max.z = half.z;
		cluster->min.z = min.z;
	}

	{
		Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
		cluster->min.x = half.x;
		cluster->max.x = max.x;
		cluster->min.y = min.y;
		cluster->max.y = half.y;
		cluster->max.z = half.z;
		cluster->min.z = min.z;
	}

	{
		Cluster::Ptr cluster = tempBuffer.emplace_back(new Cluster());
		cluster->min.x = half.x;
		cluster->max.x = max.x;
		cluster->min.y = half.y;
		cluster->max.y = max.y;
		cluster->max.z = half.z;
		cluster->min.z = min.z;
	}

	return tempBuffer;
}


void spaceTree02::DebugInfo::Clear() {
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

void spaceTree02::DebugInfo::Print() {
	printf("Clusters info.\n");
	printf("count cluster: %i count body: %i \n", countCluster, countBodies);
	printf("count level: %i minBodies: %i maxBodies: %i \n", countLevel, minBodies, maxBodies);

	printf("\nSize [%f, %f, %f]\n", size.x, size.y, size.z);
	printf("Min size [%f, %f, %f]\n", minSize.x, minSize.y, minSize.z);
	printf("Max size [%f, %f, %f]\n", maxSize.x, maxSize.y, maxSize.z);
}
