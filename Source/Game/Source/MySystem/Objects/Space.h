#pragma once

#include <string>
#include <vector>
#include <memory>
#include <glm/mat4x4.hpp>
#include "json/json.h"
#include "Math/Vector.h"
#include "MyStl/shared.h"
#include "../UI/SpatialGrid.h"
#include "Body.h"
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

	virtual Body::Ptr& Add(Body* body);
	virtual Body::Ptr& Add(Body::Ptr& body);
	virtual void RemoveBody(Body::Ptr& body);
	
	template<typename ... Args>
	Body& Add(Args&& ... args) {
		Body* body = new Body(std::forward<Args>(args)...);
		_bodies.emplace_back(body);
		return *body;
	}
	
	virtual void Preparation() {}

	std::vector<Body::Ptr>& Objects() {
		return _bodies;
	}
	
	std::pair<bool, Body&> RefFocusBody();
	Math::Vector3d CenterMass();
	Body::Ptr GetHeaviestBody();
	Body::Ptr GetBody(const char* chName);
	virtual void RemoveVelocity(bool toCenter = false);
	Body::Ptr HitObject(const glm::mat4x4& matCamera);
	
private:
	virtual std::string GetNameClass();
	
public:
	std::string _name;
	double _constGravity = 0.01f;

	float deltaTime = 1.f;
	size_t countOfIteration = 1;
	double timePassed = 0;
	bool processGPU = false;
	int tag = 0;

	std::map<std::string, std::string> _params;
	std::vector<Body::Ptr> _bodies;

	SpatialGrid spatialGrid;
	std::shared_ptr<Object> _skyboxObject;
	Body::Ptr _focusBody;
	Body::Ptr _selectBody;
	std::vector<std::pair<Body::Ptr, std::string>> _heaviestInfo;
};
