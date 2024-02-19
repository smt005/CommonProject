#include "SpaceTree00.h"
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <unordered_map>>
#include <set>
#include <Core.h>
#include "BodyData.h"

size_t Cluster::countBodiesCluster = 100;
std::vector<Cluster*> Cluster::vecClusters;

void Cluster::LoadBodies(std::vector<Body::Ptr>& bodiesPtr) {
	xMin = -1000000.f;
	xMax = 1000000.f;
	yMin = -1000000.f;
	yMax = 1000000.f;

	delete clusters[0][0];
	delete clusters[0][1];
	delete clusters[1][0];
	delete clusters[1][1];

	clusters[0][0] = nullptr;
	clusters[0][1] = nullptr;
	clusters[1][0] = nullptr;
	clusters[1][1] = nullptr;

	bodies.clear();
	bodies.reserve(bodiesPtr.size());
	
	for (Body::Ptr& bodyPtr : bodiesPtr) {
		bodies.emplace_back(bodyPtr.get());
	}

	vecClusters.clear();
}

void Cluster::CreateChilds() {
	if (bodies.size() < countBodiesCluster) {
		if (!bodies.empty()) {
			vecClusters.emplace_back(this);
		}
		return;
	}

	clusters[0][0] = new Cluster();
	clusters[0][1] = new Cluster();
	clusters[1][0] = new Cluster();
	clusters[1][1] = new Cluster();

	float xFalf = xMin + (xMax - xMin) / 2.f;
	float yFalf = yMin + (yMax - yMin) / 2.f;

	clusters[0][0]->xMin = xMin;
	clusters[0][0]->xMax = xFalf;
	clusters[0][0]->yMin = yMin;
	clusters[0][0]->yMax = yFalf;

	clusters[0][1]->xMin = xMin;
	clusters[0][1]->xMax = xFalf;
	clusters[0][1]->yMin = yFalf;
	clusters[0][1]->yMax = yMax;

	clusters[1][0]->xMin = xFalf;
	clusters[1][0]->xMax = xMax;
	clusters[1][0]->yMin = yMin;
	clusters[1][0]->yMax = yFalf;

	clusters[1][1]->xMin = yFalf;
	clusters[1][1]->xMax = yMax;
	clusters[1][1]->yMin = yFalf;
	clusters[1][1]->yMax = yMax;

	for (Body* body : bodies) {
		auto pos = body->GetPos();

		if (pos.x <= xFalf && pos.y <= yFalf) {
			clusters[0][0]->bodies.emplace_back(body);
		} else
		if (pos.x <= xFalf && pos.y > yFalf) {
			clusters[0][1]->bodies.emplace_back(body);
		} else
		if (pos.x > xFalf && pos.y <= yFalf) {
			clusters[1][0]->bodies.emplace_back(body);
		} else
		if (pos.x > xFalf && pos.y > yFalf) {
			clusters[1][1]->bodies.emplace_back(body);
		}
	}

	bodies.clear();

	clusters[0][0]->CreateChilds();
	clusters[0][1]->CreateChilds();
	clusters[1][0]->CreateChilds();
	clusters[1][1]->CreateChilds();
}

void Cluster::Print(std::string tab) {
	tab += '.';

	if (bodies.empty()) {
		if (clusters[0][0] == nullptr) {
			std::cout << tab << " [" << (int)xMin << ", " << (int)yMin << " x " << (int)xMax << ", " << (int)yMax << "] EMPTY" << std::endl;
		}
		else {
			std::cout << tab << " [" << (int)xMin << ", " << (int)yMin << " x " << (int)xMax << ", " << (int)yMax << "] NODE:" << std::endl;
			clusters[0][0]->Print(tab);
			clusters[0][1]->Print(tab);
			clusters[1][0]->Print(tab);
			clusters[1][1]->Print(tab);
		}
	}
	else {
		std::cout << tab << " [" << (int)xMin << ", " << (int)yMin << " x " << (int)xMax << ", " << (int)yMax << "] bodies:" << bodies.size() << std::endl;
	}	
}

//.............................................................
std::string SpaceTree00::GetNameClass() {
	return Engine::GetClassName(this);
}

void SpaceTree00::Update(double dt) {
	if (countOfIteration == 0 || _bodies.size() <= 1) {
		return;
	}

	for (size_t i = 0; i < countOfIteration; ++i) {
		Update();
	}
}

void SpaceTree00::Update() {
	GenerateClusters();

	{
		double time = Engine::Core::currentTime();

		UpdateForceClusters(&cluster);
		UpdateForceClusters();

		double time1 = Engine::Core::currentTime();
		double dTime = time1 - time;
		printf("Update force: %lf\n", dTime);
	}

	{
		double time = Engine::Core::currentTime();

		ApplyForce();

		double time1 = Engine::Core::currentTime();
		double dTime = time1 - time;
		printf("Update pos: %lf\n", dTime);
	}


	Preparation();
}

void SpaceTree00::UpdateForceClusters() {
	size_t size = _bodies.size();
	size_t countCluster = Cluster::vecClusters.size();

	for (size_t index = 0; index < size; ++index) {
		BodyCluster& body = *static_cast<BodyCluster*>(_bodies[index].get());
		auto pos = body.GetPos();;
		float mass = body.Mass();
		Math::Vector3& forceVec = body.Force();

		for (size_t j = 0; j < countCluster; ++j) {
			Cluster* cluster = Cluster::vecClusters[j];

			if (body.cluster == cluster) {
				continue;
			}

			Math::Vector3 gravityVec = cluster->clusterCenterMass - pos;
			double dist = Math::length(gravityVec);
			gravityVec = Math::normalize(gravityVec);

			double force = _constGravity * (cluster->clueterMass * mass) / (dist * dist);
			gravityVec *= force;
			forceVec += gravityVec;
		}
	}
}

void SpaceTree00::UpdateForceClusters(Cluster* cluster) {
	if (!cluster) {
		return;
	}

	UpdateForceClusters(cluster->clusters[0][0]);
	UpdateForceClusters(cluster->clusters[0][1]);
	UpdateForceClusters(cluster->clusters[1][0]);
	UpdateForceClusters(cluster->clusters[1][1]);

	//...
	std::vector<Body*>& bodies = cluster->bodies;
	const size_t countDatas = bodies.size();
	std::vector<BodyData::Data> datas;
	datas.resize(countDatas);

	float sumMass = 0.f;
	Math::Vector3 sumMassPos;

	for (size_t index = 0; index < countDatas; ++index) {
		BodyCluster& body = *static_cast<BodyCluster*>(bodies[index]);

		body.cluster = cluster;

		double mass = body.Mass();
		Math::Vector3 pos = body.GetPos();
		Math::Vector3& forceVec = body.Force();
		forceVec.x = 0.f;
		forceVec.y = 0.f;
		forceVec.z = 0.f;

		sumMass += mass;
		sumMassPos += (pos * mass);

		for (size_t otherIndex = 0; otherIndex < countDatas; ++otherIndex) {
			if (index == otherIndex) {
				continue;
			}

			BodyCluster& otherBody = *static_cast<BodyCluster*>(bodies[otherIndex]);

			Math::Vector3 gravityVec = otherBody.GetPos() - pos;
			double dist = Math::length(gravityVec);
			gravityVec = Math::normalize(gravityVec);

			double force = _constGravity * (mass * otherBody.Mass()) / (dist * dist);
			gravityVec *= force;
			forceVec += gravityVec;
		}
	}

	cluster->clueterMass = sumMass;
	cluster->clusterCenterMass = sumMassPos / sumMass;
}

void SpaceTree00::ApplyForce() {
	size_t size = _bodies.size();

	for (size_t index = 0; index < size; ++index) {
		BodyCluster& body = *static_cast<BodyCluster*>(_bodies[index].get());

		Math::Vector3 acceleration = body.Force() / body.Mass();
		Math::Vector3 newVelocity = acceleration * deltaTime;

		body.Velocity() += newVelocity;

		Math::Vector3 pos = body.GetPos();
		pos += body.Velocity() * deltaTime;
		body.SetPos(pos);
	}
}

void SpaceTree00::GenerateClusters() {
	double time = Engine::Core::currentTime();

	cluster.countBodiesCluster = 100;
	cluster.LoadBodies(_bodies);
	cluster.CreateChilds();

	if (lastCountBodies != _bodies.size()) {
		lastCountBodies = _bodies.size();

		for (Body::Ptr bodyPtr : _bodies) {
			bodyPtr->CalcScale();
		}
	}

	double time1 = Engine::Core::currentTime();
	double dTme = time1 - time;
	printf("Preparation:: %lf\n", dTme);
}

void SpaceTree00::Preparation() {

}
