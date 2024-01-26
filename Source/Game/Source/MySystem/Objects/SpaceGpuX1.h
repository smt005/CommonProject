#pragma once

#include "Space.h"
#include <memory>
#include <vector>
#include <json/json.h>
#include <Common/Common.h>
#include "../../CUDA/Source/Wrapper.h"

class SpaceGpuX1 final : public Space {
public:
	using Ptr = std::shared_ptr<SpaceGpuX1>;

	SpaceGpuX1() = default;
	SpaceGpuX1(const std::string& name)
		: Space(name) {
	}
	SpaceGpuX1(Json::Value& valueData)
		: Space(valueData) {
	}

	void Update(double dt) override;
	void Preparation() override;

public:
	std::string GetNameClass() override;

private:
	std::vector<CUDA::Vector3> _positions;
	std::vector<float> _masses;
	std::vector<CUDA::Vector3> _forces;
	std::vector<CUDA::Vector3> _velocities;
};
