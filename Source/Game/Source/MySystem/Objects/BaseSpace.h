#pragma once

#include "Space.h"
#include <memory>
#include "json/json.h"

class BaseSpace final : public Space {
public:
	using Ptr = std::shared_ptr<BaseSpace>;

	BaseSpace() = default;
	BaseSpace(const std::string& name)
		: Space(name) {
	}
	BaseSpace(Json::Value& valueData)
		: Space(valueData) {
	}

private:
	std::string GetNameClass() override {
		return Engine::GetClassName(this);
	}
};
