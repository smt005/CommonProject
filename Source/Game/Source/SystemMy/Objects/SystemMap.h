#pragma once

#include <string>
#include <vector>
#include <memory>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include "../../Engine/Source/Object/Model.h"

class Model;
using ValueT = float;
using Vector3 = glm::vec3;
using Matrix44 = glm::mat4x4;

class SystemMap;

class Body final  {
	friend SystemMap;

public:
	Body(std::shared_ptr<Model>& model) : _model(model) {}
	Body(const std::string& nameModel);
	Body(const std::string& nameModel, const Vector3& pos, const Vector3& velocity, ValueT mass, const std::string& name);
	~Body();

	void SetName(const std::string& name) {
		size_t size = name.size();
		_name = new char[size + 1];
		memcpy(_name, name.data(), size);
		_name[size] = '\0';
	}

	Vector3 GetPos() const { 
		return Vector3(_matrix[3][0], _matrix[3][1], _matrix[3][2]);
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
	
	const Matrix44& getMatrix() const {
		return _matrix;
	}

private:
public:
	char* _name = nullptr;
	ValueT _mass = 0;
	Vector3 _velocity = { 0, 0, 0 };
	Vector3 _force = { 0, 0, 0 };
	Matrix44 _matrix = Matrix44(1);
	std::shared_ptr<Model> _model;
};

class SystemMap final {
public:
	using Ptr = std::shared_ptr<SystemMap>;

	SystemMap(const std::string& name) : _name(name) {}
	~SystemMap();

	void Update(double dt);
	void Save();
	bool Load();

	Vector3 CenterMass();
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

private:
public:
	ValueT _constGravity = 0.01f;
	std::string _name;
	std::vector<Body*> _bodies;
};
