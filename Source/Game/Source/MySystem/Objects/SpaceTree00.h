#pragma once

#include "Space.h"
#include <memory>
#include <vector>
#include <json/json.h>
#include <Common/Common.h>
#include <Wrapper.h>
#include <Classes.h>
#include "BodyData.h"

struct Cluster {
	Cluster() {
		clusters[0][0] = nullptr;
		clusters[0][1] = nullptr;
		clusters[1][0] = nullptr;
		clusters[1][1] = nullptr;
	}
	~Cluster() {
		delete clusters[0][0];
		delete clusters[0][1];
		delete clusters[1][0];
		delete clusters[1][1];
	}

	float xMin, xMax, yMin, yMax;
	Cluster* clusters[2][2];
	std::vector<Body*> bodies;
	float clueterMass = 0.f;
	Math::Vector3 clusterCenterMass;

	void LoadBodies(std::vector<Body::Ptr>& bodies);
	void CreateChilds();
	void Print(std::string tab);

	static size_t countBodiesCluster;
	static std::vector<Cluster*> vecClusters;
	std::string tab;
};

class BodyCluster final : public BodyData {
public:
	Cluster* cluster = nullptr;
};

class SpaceTree00 final : public Space {
private:
	

public:
	using Ptr = std::shared_ptr<SpaceTree00>;

	SpaceTree00() = default;
	SpaceTree00(const std::string& name)
		: Space(name) {
	}
	SpaceTree00(Json::Value& valueData)
		: Space(valueData) {
	}

	void Update(double dt) override;
	void Preparation() override;

private:
	void Update();
	void UpdateForceClusters();
	void UpdateForceClusters(Cluster* cluster);
	void ApplyForce();

public:
	std::string GetNameClass() override;

private:
	Cluster cluster;
};
