#pragma once

#include "Space.h"
#include <memory>

class BaseSpace final : public Space {
public:
	using Ptr = std::shared_ptr<BaseSpace>;
	BaseSpace(const std::string& name)
		: Space(name) {
	}
};
