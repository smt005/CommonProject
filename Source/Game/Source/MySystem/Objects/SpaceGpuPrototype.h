#pragma once

#include "Space.h"
#include <memory>
#include <json/json.h>
#include <Common/Common.h>

class SpaceGpuPrototype final : public Space {
public:
	using Ptr = std::shared_ptr<SpaceGpuPrototype>;

	SpaceGpuPrototype() = default;
	SpaceGpuPrototype(const std::string& name)
		: Space(name) {
	}
	SpaceGpuPrototype(Json::Value& valueData)
		: Space(valueData) {
		if (_params["PROCESS"] == "GPU") {
			processGPU = true;
		}
	}

	void Update(double dt) override;

private:
	std::string GetNameClass() override {
		return Engine::GetClassName(this);
	}

private:
	bool processGPU = false;
};
