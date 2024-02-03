#pragma once

#include <string>
#include <vector>
#include <glm/mat4x4.hpp>
#include "Math/Vector.h"
#include "MyStl/shared.h"
#include <memory>
#include "../../Engine/Source/Object/Model.h"
#include "Body.h"

class Space;
class Model;
using ModelPtr = std::shared_ptr<Model>;

class BodyData final : public Body {
	friend Space;

public:
	using Ptr = mystd::shared<BodyData>;

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

	BodyData() = default;
	BodyData(std::shared_ptr<Model>& model) : _model(model) {}
	BodyData(const std::string& nameModel);
	BodyData(const std::string& nameModel, const Math::Vector3d& pos, const Math::Vector3d& velocity, double mass, const std::string& name);

	Math::Vector3d GetPos() const {
		return Math::Vector3d(_matrix[3][0], _matrix[3][1], _matrix[3][2]);
	}

	void SetPos(const Math::Vector3d& pos) {
		_matrix[3][0] = pos[0];
		_matrix[3][1] = pos[1];
		_matrix[3][2] = pos[2];

		if (_dataPtr) {
			_dataPtr->pos = pos;
		}
	}

	void SetVelocity(const Math::Vector3d& velocity) {
		_velocity = velocity;
	}

	bool GetModel() const {
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
	double _mass = 0;
	Math::Vector3d _force;
	Math::Vector3d _velocity;
	float _angular = 0.f;
	float _angularVelocity = -0.0005f;
	float _scale = 1.f;

	glm::mat4x4 _matrix = glm::mat4x4(1);
	std::shared_ptr<Model> _model;
	Data* _dataPtr = nullptr;

public:
	double force = 0.0;
};
