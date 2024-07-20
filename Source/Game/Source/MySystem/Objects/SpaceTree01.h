// ◦ Xyz ◦
#pragma once

#include "Space.h"
#include <memory>
#include <vector>
#include <json/json.h>
#include <Common/Common.h>
#include <Wrapper.h>
#include <Classes.h>
#include "BodyData.h"

namespace spaceTree01
{
	struct DebugInfo
	{
		size_t countLevel = 0;
		size_t countCluster = 0;
		size_t countBodies = 0;
		size_t minBodies = std::numeric_limits<float>::max();
		size_t maxBodies = 0;

		Math::Vector3 min = Math::Vector3(std::numeric_limits<float>::max());
		Math::Vector3 max;
		Math::Vector3 size;
		Math::Vector3 minSize = Math::Vector3(std::numeric_limits<float>::max());
		Math::Vector3 maxSize;

		void Clear();
		void Print();
	};

	struct Cluster01
	{
		//using Ptr = mystd::shared<Cluster01>;
		using Ptr = std::shared_ptr<Cluster01>;

		Math::Vector3 min;
		Math::Vector3 max;

		Math::Vector3 centerPos;
		Math::Vector3 centerMass;
		Math::Vector3 force;
		float mass = 0;
		float dist = 0;

		std::vector<Body*> bodies;
	};
}

class SpaceTree01 final : public Space
{
public:
	using Ptr = std::shared_ptr<SpaceTree01>;

	SpaceTree01() = default;
	SpaceTree01(const std::string& name)
		: Space(name)
	{
		LoadProperty();
	}
	SpaceTree01(Json::Value& valueData)
		: Space(valueData)
	{
		LoadProperty();
	}

	void Update(double dt) override;
	void Preparation() override;

private:
	void Update();
	void UpdateForceByBody(std::vector<Body*>& bodies, std::vector<Body*>& subBodies);
	void UpdateForceByCluster(spaceTree01::Cluster01& cluster);
	bool IsClosestCluster(spaceTree01::Cluster01& cluster0, spaceTree01::Cluster01& cluster1);
	void UpdatePos();
	void GenerateClusters();
	void LoadProperty();

public:
	std::string GetNameClass() override;

private:
	std::vector<spaceTree01::Cluster01::Ptr> buffer;
	spaceTree01::DebugInfo _debugInfo;
	size_t maxCountBodies = 500;
	float distFactor = 1.f;
	bool showDebugInfo = false;
};
