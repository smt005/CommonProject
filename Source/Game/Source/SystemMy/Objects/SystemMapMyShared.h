
#pragma once
#include "../Objects/SystemClass.h"

#if SYSTEM_MAP == 8

#include <string>
#include <vector>
#include <memory>
#include <glm/mat4x4.hpp>
#include "Math/Vector.h"
#include "MyStl/shared.h"
#include "../UI/SpatialGrid.h"

#include "../../Engine/Source/Object/Model.h"
#include "../../Engine/Source/Object/Object.h"

class SystemMap;

class Body final  {
	friend SystemMap;

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

#endif
