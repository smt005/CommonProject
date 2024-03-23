// ◦ Xyz ◦
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

				// Создание прямоугольников
				cluster->boxPtr = std::make_shared<Box>(cluster->min, cluster->max);

				// Color
				float r = 0.f;
				float g = 0.f;
				float b = 0.f;
				float a = 0.1f;

				{
					r = ((cluster->max.x + _debugInfo.min.x) * 0.5f - _debugInfo.min.x) / _debugInfo.size.x;
					r = r < 0.f ? 0.f : r;
					r = r < 1.f ? r : 1.f;
				}
				{
					g = ((cluster->max.y + _debugInfo.min.y) * 0.5f - _debugInfo.min.y) / _debugInfo.size.y;
					g = g < 0.f ? 0.f : g;
					g = g > 1.f ? 1.f : g;
				}
				{
					b = ((cluster->max.z + _debugInfo.min.z) * 0.5f - _debugInfo.min.z) / _debugInfo.size.z;
					b = b < 0.f ? 0.f : b;
					b = b > 1.f ? 1.f : b;
				}
				a = 0.25f;

				cluster->boxPtr->setColor(Color(r, g, b, a));

				continue;
			}

			//...
			Math::Vector3 min = cluster->min;
			Math::Vector3 max = cluster->max;
			Math::Vector3 half = min + (max - min) / 2.f;

			std::vector<Cluster::Ptr> tempBuffer = CreateSubClusters(min, max, half);

			//...
			for (Body* body : cluster->bodies) {
				Math::Vector3 pos = body->GetPos();

				for (auto& cl : tempBuffer) {
					if (cl->IsInside(pos)) {
						cl->bodies.emplace_back(body);
						break;
					}
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
	if (false) {
		float minX = 0.f;
		float minY = 0.f;
		float minZ = 0.f;
		float maxX = 0.f;
		float maxY = 0.f;
		float maxZ = 0.f;

		for (auto& cl : buffer) {
			minX = cl->min.x < minX ? cl->min.x : minX;
			minY = cl->min.y < minY ? cl->min.y : minY;
			minZ = cl->min.z < minZ ? cl->min.z : minZ;

			maxX = cl->max.x > maxX ? cl->max.x : maxX;
			maxY = cl->max.y > maxY ? cl->max.y : maxY;
			maxZ = cl->max.z > maxZ ? cl->max.z : maxZ;
		}

		printf("Box: min: [%f, %f, %f] max: [%f, %f, %f]\n", minX, minY, minZ, maxX, maxY, maxZ);
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


bool spaceTree02::Cluster02::IsInside(const Math::Vector3& pos) {
	if (pos.x > min.x && pos.x <= max.x &&
		pos.y > min.y && pos.y <= max.y &&
		pos.z > min.z && pos.z <= max.z)
	{
		return true;
	}
	return false;
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
