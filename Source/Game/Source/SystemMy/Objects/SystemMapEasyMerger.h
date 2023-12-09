
#pragma once
#include "../Objects/SystemClass.h"

#if SYSTEM_MAP == 6

#include <string>
#include <vector>
#include <memory>
#include <glm/mat4x4.hpp>
#include "Math/Vector.h"
#include "../UI/SpatialGrid.h"
#include "../../Engine/Source/Object/Model.h"

namespace MAP_EASY_MERGE {
class SystemMap;

class Body final  {
	friend SystemMap;

public:
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

private:
public:
	char* _name = nullptr;
	double _mass = 0;
	Math::Vector3d _velocity;

	glm::mat4x4 _matrix = glm::mat4x4(1);
	std::shared_ptr<Model> _model;
	Data* _dataPtr = nullptr;
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

	Math::Vector3d CenterMass();
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

	Body* GetHeaviestBody(bool setAsStar = true);

	bool CHECK();

public:
	SpatialGrid spatialGrid;
	int time = 0;
	bool threadEnable = true;

private:
public:
	double _constGravity = 0.01f;
	std::string _name;
	std::vector<Body*> _bodies;
	std::vector<Body::Data> _datas;
};

}

#endif
