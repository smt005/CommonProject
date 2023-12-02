
#pragma once
#include "../Objects/SystemClass.h"

#if SYSTEM_MAP == 3

#include <string>
#include <vector>
#include <memory>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include "../UI/SpatialGrid.h"
#include "../../Engine/Source/Object/Model.h"

namespace STATIC_ARR {
class SystemMap;

class Body final  {
	friend SystemMap;

public:
	struct Data {
		float mass;
		glm::vec3 pos;
		glm::vec3 force;

		Data() = default;
		Data(const float _mass, glm::vec3&& _pos)
			: mass(_mass)
			, pos(_pos)
			, force(0, 0, 0)
		{}
	};

	Body() = default;
	Body(std::shared_ptr<Model>& model) : _model(model) {}
	Body(const std::string& nameModel);
	Body(const std::string& nameModel, const glm::vec3& pos, const glm::vec3& velocity, float mass, const std::string& name);
	~Body();

	void SetName(const std::string& name) {
		size_t size = name.size();
		_name = new char[size + 1];
		memcpy(_name, name.data(), size);
		_name[size] = '\0';
	}

	glm::vec3 GetPos() const { 
		return glm::vec3(_matrix[3][0], _matrix[3][1], _matrix[3][2]);
	}

	template<typename T>
	void SetPos(T&& pos) {
		_matrix[3][0] = pos[0];
		_matrix[3][1] = pos[1];
		_matrix[3][2] = pos[2];

		if (_dataPtr) {
			_dataPtr->pos = pos;
		}
	}

	const char* const Name() const {
		return _name;
	}

	template<typename T>
	void SetVelocity(T&& velocity) {
		_velocity = velocity;
	}

	Model& getModel() {
		return *_model;
	}
	
	const glm::mat4x4& getMatrix() const {
		return _matrix;
	}

private:
public:
	char* _name = nullptr;
	float _mass = 0;
	glm::vec3 _velocity = { 0, 0, 0 };

	glm::mat4x4 _matrix = glm::mat4x4(1);
	std::shared_ptr<Model> _model;
	Data* _dataPtr = nullptr;
};

struct SystemStackData {
	SystemStackData() = default;

	size_t size = 0;
	size_t capacity = 2000;
	Body::Data bodies[2000];

	static SystemStackData data;
};

class SystemMap final {
public:
	using Ptr = std::shared_ptr<SystemMap>;

	SystemMap(const std::string& name);
	~SystemMap();

	void Update(double dt, int countForceTime);
	void Update(double dt);
	void Save();
	bool Load();

	glm::vec3 CenterMass();
	Body* GetBody(const char* chName);

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

	std::vector<Body*>& Objects() {
		return _bodies;
	}

	void DataAssociation();

public:
	SpatialGrid spatialGrid;
	int time = 0;
	bool threadEnable = true;

private:
public:
	float _constGravity = 0.01f;
	std::string _name;
	std::vector<Body*> _bodies;
};

}

#endif
