#pragma once

#include "Space.h"
#include <memory>
#include <json/json.h>
#include <Common/Common.h>

class SpaceGpuPrototypeV3 final : public Space {
public:
	using Ptr = std::shared_ptr<SpaceGpuPrototypeV3>;

	SpaceGpuPrototypeV3() = default;
	SpaceGpuPrototypeV3(const std::string& name)
		: Space(name) {
	}
	SpaceGpuPrototypeV3(Json::Value& valueData)
		: Space(valueData) {
		if (_params["PROCESS"] == "GPU") {
			processGPU = true;
		}
		tag = atoi(_params["TAG"].c_str());
	}

	void Update(double dt) override;

private:
	std::string GetNameClass() override {
		return Engine::GetClassName(this);
	}

private:
public:
	bool processGPU = false;
	int tag = 0;
};
