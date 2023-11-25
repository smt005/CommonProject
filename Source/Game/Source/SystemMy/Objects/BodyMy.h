#pragma once

#include "Object/Object.h"
#include "glm/vec3.hpp"
#include "Object/Map.h"
#include "Object/Line.h"
#include <list>
#include "glm/vec3.hpp"

class BodyMy final : public Object {
public:
	typedef std::shared_ptr<BodyMy> Ptr;

	struct ForceData {
		ForceData(BodyMy* _body, const glm::vec3& _forceVec3)
			: body(_body)
			, forceVec3(_forceVec3)
		{}
		BodyMy* body = nullptr;
		vec3 forceVec3;
	};

	BodyMy(const string& name, const string& modelName, const vec3& pos)
		: Object( name, modelName, pos, Engine::Physics::Type::CONVEX)
	{
		tag = 123;
	}

	void action() override;

	glm::vec3 GetVector();
	Line& LineToCenter();
	Line& LineToMassCenter();

	Line& LineForceVector() {
		return _forceVector;
	}
	Line& LinePath() {
		return _path;
	}
	Line& RelativelyLinePath() {
		return _relativelyPath;
	}
	
	void CalculateRelativelyLinePath();


	void AddForceMy(const glm::vec3& forceVec);
	glm::vec3 _velocity = { 0.f, 0.f, 0.f };

	float speed = 0;
	float minSpeed = 0;
	float maxSpeed = 0;
	double time = 0;

private:
	Line _lineToCenter;
	Line _lineToMassCenter;
	Line _forceVector;
	Line _path;
	Line _relativelyPath;

	std::list<glm::vec3> _points;

public:
	static glm::vec3 CenterSystem();
	static glm::vec3 CenterMassSystem();
	static void UpdateRalatovePos();
	static void RemoveBody();

public:
	static float G;
	static Map::Ptr system;
	static glm::vec3 centerSystem;
	static glm::vec3 centerMassSystem;
	static std::vector<Object*> removeObject;
	static BodyMy* centerBody;

	static std::vector<ForceData> bodyes;
	static void ApplyForce(const double dt);
};
