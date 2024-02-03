#pragma once

#include "Space.h"
#include "BodyData.h"

class BaseSpace : public Space {
public:
	using Ptr = std::shared_ptr<BaseSpace>;

	BaseSpace() = default;
	BaseSpace(const std::string& name)
		: Space(name) {
	}
	BaseSpace(Json::Value& valueData)
		: Space(valueData) {
	}
	virtual ~BaseSpace() = default;

	void Update(double dt) override;
	void Preparation() override;

public:
	std::string GetNameClass() override;
	void Update();

private:
	std::vector<BodyData::Data> _datas;
};
