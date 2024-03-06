#pragma once

#include "Space.h"
#include <memory>
#include <vector>
#include <json/json.h>
#include <Common/Common.h>
#include <Wrapper.h>
#include <Classes.h>
#include "BodyData.h"
#include "../../Engine/Source/Object/Samples/Box.h"

namespace spaceTree02 {
	struct DebugInfo {
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

	struct Cluster02 {
		//using Ptr = mystd::shared<Cluster02>;
		using Ptr = std::shared_ptr<Cluster02>;

		Math::Vector3 min;
		Math::Vector3 max;
		Math::Vector3 size;

		Math::Vector3 centerPos;
		Math::Vector3 centerMass;
		Math::Vector3 force;
		float mass = 0;
		float dist = 0;

		Box::Ptr boxPtr;

		std::vector<Body*> bodies;

		bool IsInside(const Math::Vector3& pos);
	};
}

class SpaceTree02 final : public Space {
public:
	using Ptr = std::shared_ptr<SpaceTree02>;

	SpaceTree02() = default;
	SpaceTree02(const std::string& name)
		: Space(name) {
		LoadProperty();
	}
	SpaceTree02(Json::Value& valueData)
		: Space(valueData) {
		LoadProperty();
	}

	void Update(double dt) override;
	void Preparation() override;

private:
	void Update();
	void UpdateForceByBody(std::vector<Body*>& bodies, std::vector<Body*>& subBodies);
	void UpdateForceByCluster(spaceTree02::Cluster02& cluster);
	bool IsClosestCluster(spaceTree02::Cluster02& cluster0, spaceTree02::Cluster02& cluster1);
	void UpdatePos();
	void GenerateClusters();
	std::vector<spaceTree02::Cluster02::Ptr> CreateSubClusters(const Math::Vector3& min, const Math::Vector3& max, const Math::Vector3& half);
	void LoadProperty();

public:
	std::string GetNameClass() override;

public:
	bool searchOptimalCountBodies = true;

private:
public:
	std::vector<spaceTree02::Cluster02::Ptr> buffer;
	spaceTree02::DebugInfo _debugInfo;
	
	float distFactor = 1.f;
	bool showDebugInfo = false;

	size_t minCountOperation = std::numeric_limits<size_t>::max();
	size_t optimalCountBodies = 10;
	size_t maxCountBodies = 10;
	size_t minOptimalCountBodies = 10;
	size_t maxOptimalCountBodies = 200;
	size_t deltaOptimalCountBodies = 5;
};
