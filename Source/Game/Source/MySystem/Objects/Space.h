#pragma once

#include <string>
#include <vector>
#include <memory>
#include <glm/mat4x4.hpp>
#include "json/json.h"
#include "Math/Vector.h"
#include "MyStl/shared.h"
#include "../UI/SpatialGrid.h"
#include "BodyData.h"
#include "Object/Object.h"

class Space {
public:
	using Ptr = std::shared_ptr<Space>;

	Space() = default;
	Space(const std::string& name);
	Space(Json::Value& valueData);
	virtual ~Space() = default;
	
	virtual void Update(double dt) {}
	virtual void Save();
	virtual bool Load(Json::Value& valueData);
	virtual bool Load();

	virtual BodyData::Ptr& Add(BodyData* body);
	virtual BodyData::Ptr& Add(BodyData::Ptr& body);
	virtual void RemoveBody(BodyData::Ptr& body);
	
	template<typename ... Args>
	BodyData& Add(Args&& ... args) {
		BodyData* body = new BodyData(std::forward<Args>(args)...);
		_bodies.emplace_back(body);
		return *body;
	}
	
	virtual void Preparation() {}

	std::vector<BodyData::Ptr>& Objects() {
		return _bodies;
	}
	
	std::pair<bool, BodyData&> RefFocusBody();
	Math::Vector3d CenterMass();
	BodyData::Ptr GetHeaviestBody();
	BodyData::Ptr GetBody(const char* chName);
	virtual void RemoveVelocity(bool toCenter = false);
	BodyData::Ptr HitObject(const glm::mat4x4& matCamera);
	
public:
	virtual std::string GetNameClass();
	
public:
	std::string _name;
	double _constGravity = 0.01f;

	float deltaTime = 1.f;
	size_t countOfIteration = 1;
	double timePassed = 0;
	bool processGPU = false;
	bool multithread = false;
	int tag = 0;

	std::map<std::string, std::string> _params;
	std::vector<BodyData::Ptr> _bodies;

	SpatialGrid spatialGrid;
	std::shared_ptr<Object> _skyboxObject;
	BodyData::Ptr _focusBody;
	BodyData::Ptr _selectBody;
	std::vector<std::pair<BodyData::Ptr, std::string>> _heaviestInfo;
};
