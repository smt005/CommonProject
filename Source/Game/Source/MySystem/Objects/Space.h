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

class Space final {
public:
	using Ptr = std::shared_ptr<Space>;

	Space(const std::string& name);
	~Space();


	void Update() {
		Update(deltaTime, countOfIteration);
	}
	void Update(double dt, int countForceTime);
	void Update(double dt);
	void Save();
	bool Load();

	Math::Vector3d CenterMass();
	Body::Ptr GetBody(const char* chName);

	template<typename ... Args>
	Body& Add(Args&& ... args) {
		Body* body = new Body(std::forward<Args>(args)...);
		_bodies.emplace_back(body);
		DataAssociation(); // TODO:
		return *body;
	}

	Body& Add(Body* body) {
		_bodies.emplace_back(body);
		DataAssociation(); // TODO:
		return *body;
	}

	void RemoveBody(Body::Ptr& body);

	std::vector<Body::Ptr>& Objects() {
		return _bodies;
	}

	void DataAssociation();

	Body::Ptr GetHeaviestBody(bool setAsStar = true);
	void RemoveVelocity(bool toCenter = false);

	Body& RefFocusBody() {
		auto it = std::find(_bodies.begin(), _bodies.end(), _focusBody);
		if (it != _bodies.end()) {
			return **it;
		}

		static Body defaultBody;
		return defaultBody;
	}

	Body::Ptr HitObject(const glm::mat4x4& matCamera);

	bool CHECK();

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
