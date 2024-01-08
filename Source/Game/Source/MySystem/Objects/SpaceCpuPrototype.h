#pragma once

#include "Space.h"
#include <memory>
#include <json/json.h>
#include <Common/Common.h>

class SpaceCpuPrototype final : public Space {
public:
	using Ptr = std::shared_ptr<SpaceCpuPrototype>;

	SpaceCpuPrototype() = default;
	SpaceCpuPrototype(const std::string& name)
		: Space(name) {
	}
	SpaceCpuPrototype(Json::Value& valueData)
		: Space(valueData) {
	}

	void Update(double dt) override;

private:
	std::string GetNameClass() override {
		return Engine::GetClassName(this);
	}
};
