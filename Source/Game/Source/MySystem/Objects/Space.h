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

class Space {
public:
	using Ptr = std::shared_ptr<Space>;

	Space(const std::string& name);
	virtual ~Space();

	virtual void Update() {
		Update(deltaTime, countOfIteration);
	}
	virtual void Update(double dt, int countForceTime);
	virtual void Update(double dt);
	virtual void Save();
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
