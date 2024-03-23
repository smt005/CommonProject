// ◦ Xyz ◦
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

class BodyData : public Body {
	friend Space;

public:
	using Ptr = mystd::shared<BodyData>;
	//using Ptr = std::shared_ptr<BodyData>;

	struct Data {
		double mass;
		Math::Vector3 pos;
		Math::Vector3 force;

		Data() = default;
		Data(const double _mass, Math::Vector3&& _pos)
			: mass(_mass)
			, pos(_pos)
			, force(0, 0, 0)
		{}
		Data(const double _mass, const Math::Vector3& _pos, const Math::Vector3& _force)
			: mass(_mass)
			, pos(_pos)
			, force(_force)
		{}
	};

	BodyData() = default;
	BodyData(std::shared_ptr<Model>& model) : _model(model) {}
	BodyData(const std::string& nameModel);
	BodyData(const std::string& nameModel, const Math::Vector3& pos, const Math::Vector3& velocity, double mass, const std::string& name);
	virtual ~BodyData() = default;

	Math::Vector3 GetPos() const {
		return Math::Vector3(_matrix[3][0], _matrix[3][1], _matrix[3][2]);
	}

	void SetPos(const Math::Vector3& pos) {
		_matrix[3][0] = pos[0];
		_matrix[3][1] = pos[1];
		_matrix[3][2] = pos[2];

		if (_dataPtr) {
			_dataPtr->pos = pos;
		}
	}

	void SetVelocity(const Math::Vector3& velocity) {
		_velocity = velocity;
	}

	bool HasModel() const {
		return _model ? true : false;
	}

	Model& getModel() override {
		return *_model;
	}
	
	const glm::mat4x4& getMatrix() const {
		return _matrix;
	}

	Math::Vector3 PosOnScreen(const glm::mat4x4& matCamera, bool applySizeScreen);
	bool hit(const glm::mat4x4& matCamera);

	void Rotate();
	void CalcScale() override ;

	Math::Vector3& Velocity() override { return _velocity; }
	Math::Vector3& Force() override { return _force; }
	float& Mass() override { return _mass; }
	float& Scale() override { return _scale; }

private:
public:
	std::string className;
	float _mass = 0;
	Math::Vector3 _force;
	Math::Vector3 _velocity;
	float _angular = 0.f;
	float _angularVelocity = -0.0005f;
	float _scale = 1.f;

	glm::mat4x4 _matrix = glm::mat4x4(1);
	std::shared_ptr<Model> _model;
	Data* _dataPtr = nullptr;

public:
	double force = 0.0;
};
