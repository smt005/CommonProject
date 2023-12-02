
#pragma once
#include "../Objects/SystemClass.h"

#if SYSTEM_MAP == 0

#include <string>
#include <vector>
#include <memory>

#include "SystemTypes.h"
#include "../UI/SpatialGrid.h"
#include "../../Engine/Source/Object/Model.h"

namespace S00 {
class SystemMap;

class Body final  {
	friend SystemMap;

public:
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
	glm::vec3 _force = { 0, 0, 0 };
	glm::mat4x4 _matrix = glm::mat4x4(1);
	std::shared_ptr<Model> _model;
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
		return *_bodies.emplace_back(body);
	}

	Body& Add(Body* body) {
		return *_bodies.emplace_back(body);
	}

	std::vector<Body*>& Objects() {
		return _bodies;
	}

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
