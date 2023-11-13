#pragma once

#include "Object/Object.h"
#include "glm/vec3.hpp"
#include "Object/Map.h"
#include "Object/Line.h"

class Body final : public Object {
public:
	typedef std::shared_ptr<Body> Ptr;

	Body(const string& name, const string& modelName, const vec3& pos)
		: Object( name, modelName, pos, Engine::Physics::Type::CONVEX)
	{
		tag = 123;
	}

	void action() override;

	glm::vec3 GetVector();
	const Line& LineToCenter();
	const Line& LineToMassCenter();

private:
	Line _lineToCenter;
	Line _lineToMassCenter;

public:
	static glm::vec3 CenterSystem();
	static glm::vec3 CenterMassSystem();
	static void RemoveBody();

public:
	static Map::Ptr system;
	static glm::vec3 centerSystem;
	static glm::vec3 centerMassSystem;
	static std::vector<Object*> removeObject;
};
