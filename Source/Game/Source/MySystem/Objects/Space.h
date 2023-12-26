#pragma once

#include <string>
#include <vector>
#include <glm/mat4x4.hpp>
#include "Math/Vector.h"
#include <memory>
#include "MyStl/shared.h"
#include "../UI/SpatialGrid.h"
#include "Body.h"
#include "Object/Object.h"
#include "json/json.h"

class Space {
public:
	using Ptr = std::shared_ptr<Space>;

	Space() = default;
	Space(const std::string& name);
	Space(Json::Value& valueData);
	virtual ~Space();

	virtual void Update() {
		Update(deltaTime, countOfIteration);
	}
	virtual void Update(double dt, int countForceTime);
	virtual void Update(double dt);
	virtual void Save();
	virtual bool Load(Json::Value& valueData);
	virtual bool Load();

	virtual Math::Vector3d CenterMass();
	virtual Body::Ptr GetBody(const char* chName);

	template<typename ... Args>
	Body& Add(Args&& ... args) {
		Body* body = new Body(std::forward<Args>(args)...);
		_bodies.emplace_back(body);
		DataAssociation(); // TODO:
		return *body;
	}

	template<typename ... Args>
	Body& AddWithoutAssociation(Args&& ... args) {
		Body* body = new Body(std::forward<Args>(args)...);
		_bodies.emplace_back(body);
		return *body;
	}

	virtual Body& Add(Body* body) {
		_bodies.emplace_back(body);
		DataAssociation(); // TODO:
		return *body;
	}

	virtual void RemoveBody(Body::Ptr& body);

	virtual std::vector<Body::Ptr>& Objects() {
		return _bodies;
	}

	virtual void DataAssociation();

	virtual Body::Ptr GetHeaviestBody(bool setAsStar = true);
	virtual void RemoveVelocity(bool toCenter = false);

	virtual Body& RefFocusBody() {
		auto it = std::find(_bodies.begin(), _bodies.end(), _focusBody);
		if (it != _bodies.end()) {
			return **it;
		}

		static Body defaultBody;
		return defaultBody;
	}

	virtual Body::Ptr HitObject(const glm::mat4x4& matCamera);

private:
	virtual std::string GetNameClass();

public:
	double deltaTime = 1;
	size_t countOfIteration = 1;
	double timePassed = 0;

	SpatialGrid spatialGrid;
	int time = 0;
	bool threadEnable = true;

	Body::Ptr _focusBody;
	Body::Ptr _selectBody;
	std::vector<std::pair<Body::Ptr, std::string>> _heaviestInfo;

	//std::shared_ptr<Model> _skyboxModel;
	std::shared_ptr<Object> _skyboxObject;

private:
public:
	double _constGravity = 0.01f;
	std::string _name;
	std::vector<Body::Ptr> _bodies;
	std::vector<Body::Data> _datas;
};

// ÿÙ·ÎÓÌ
/*
#pragma once

#include "Space.h"
#include <memory>
#include "json/json.h"

class SpaceXXX final : public Space {
public:
	using Ptr = std::shared_ptr<SpaceXXX>;

	SpaceXXX() = default;
	SpaceXXX(const std::string& name)
		: Space(name) {
	}
	SpaceXXX(Json::Value& valueData)
		: Space(valueData) {
	}

private:
	std::string GetNameClass() override {
		return Engine::GetClassName(this);
	}
};
*/
