#pragma once

#include <string>
#include <vector>
#include <memory>
#include <glm/mat4x4.hpp>
#include "Math/Vector.h"
#include "MyStl/shared.h"
#include <memory>
#include "../../Engine/Source/Object/Model.h"

class Space;
class Model;
using ModelPtr = std::shared_ptr<Model>;

class Body final  {
	friend Space;

public:
	using Ptr = mystd::shared<Body>;
	//using Ptr = std::shared_ptr<Body>;

	struct Data {
		double mass;
		Math::Vector3d pos;
		Math::Vector3d force;

		Data() = default;
		Data(const double _mass, Math::Vector3d&& _pos)
			: mass(_mass)
			, pos(_pos)
			, force(0, 0, 0)
		{}
		Data(const double _mass, const Math::Vector3d& _pos, const Math::Vector3d& _force)
			: mass(_mass)
			, pos(_pos)
			, force(_force)
		{}
	};

	Body() = default;
	Body(std::shared_ptr<Model>& model) : _model(model) {}
	Body(const std::string& nameModel);
	Body(const std::string& nameModel, const Math::Vector3d& pos, const Math::Vector3d& velocity, double mass, const std::string& name);
	~Body();

	void SetName(const std::string& name) {
		size_t size = name.size();
		_name = new char[size + 1];
		memcpy(_name, name.data(), size);
		_name[size] = '\0';
	}

	Math::Vector3d GetPos() const {
		return Math::Vector3d(_matrix[3][0], _matrix[3][1], _matrix[3][2]);
	}

	template<typename T>
	void SetPos(T&& pos) {
		_matrix = glm::mat4x4(1);
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

	bool HetModel() const {
		return _model ? true : false;
	}

	Model& getModel() {
		return *_model;
	}
	
	const glm::mat4x4& getMatrix() const {
		return _matrix;
	}

	Math::Vector3 PosOnScreen(const glm::mat4x4& matCamera, bool applySizeScreen);
	bool hit(const glm::mat4x4& matCamera);

	void Rotate();
	void Scale();

private:
public:
	std::string className;
	char* _name = nullptr;
	double _mass = 0;
	Math::Vector3d _velocity;
	float _angular = 0.f;
	float _angularVelocity = -0.0005f;
	float _scale = 1.f;

	glm::mat4x4 _matrix = glm::mat4x4(1);
	std::shared_ptr<Model> _model;
	Data* _dataPtr = nullptr;
};
