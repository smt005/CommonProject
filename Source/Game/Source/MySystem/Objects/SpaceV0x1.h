#pragma once

#include "Space.h"
#include <memory>
#include <json/json.h>
#include <Common/Common.h>

class SpaceV0x1 final : public Space {
public:
	using Ptr = std::shared_ptr<SpaceV0x1>;

	SpaceV0x1() = default;
	SpaceV0x1(const std::string& name)
		: Space(name) {
	}
	SpaceV0x1(Json::Value& valueData)
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
