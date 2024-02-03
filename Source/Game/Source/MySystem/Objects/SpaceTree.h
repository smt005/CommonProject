#pragma once

#include "Space.h"
#include <memory>
#include <vector>
#include <json/json.h>
#include <Common/Common.h>
#include <Wrapper.h>
#include <Classes.h>

class SpaceTree final : public Space {
public:
	using Ptr = std::shared_ptr<SpaceTree>;

	SpaceTree() = default;
	SpaceTree(const std::string& name)
		: Space(name) {
	}
	SpaceTree(Json::Value& valueData)
		: Space(valueData) {
	}

	void Update(double dt) override;
	void Preparation() override;

private:
	void Update();

public:
	std::string GetNameClass() override;

private:
};
