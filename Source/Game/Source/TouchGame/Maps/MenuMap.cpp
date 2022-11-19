
#include "TouchGame/Maps/MenuMap.h"
#include "Core.h"

// Puck
void MenuMap::Puck::action() {
	if (!enableForce || !MenuMap::target) {
		return;
	}

	glm::vec3 posTarget = MenuMap::target->getPos();
	glm::vec3 forceVec3 = getPos() - posTarget;

	forceVec3 = glm::normalize(forceVec3);
	forceVec3 *= _force;
	addForce(forceVec3);
}

// Target
void MenuMap::Target::action() {
	if (!enableForce || !MenuMap::target) {
		return;
	}

	glm::vec3 posTarget{ 0.f, 0.f, 1.f };
	posTarget[0] = std::cos(_angle) * _dist;
	posTarget[1] = std::sin(_angle) * _dist;
	_angle = _angle + _dAngle;

	glm::vec3 forceVec3 = getPos() - posTarget;

	forceVec3 = glm::normalize(forceVec3);
	forceVec3 *= _force;
	addForce(forceVec3);
}

// MenuMap
MenuMap::Target::Ptr MenuMap::target;
bool MenuMap::enableForce = false;

MenuMap::MenuMap() {
	Engine::Core::log("MenuMap::MenuMap() ");
};

bool MenuMap::create(const string& name) {
	if (!Map::create(name)) {
		return false;
	}

	// Puck
	for (size_t index = 0; index < 2; ++index) {
		const string name = "Puck_"s + std::to_string(index);
		const string modelName = "PuckWhite";
		const vec3 pos{ (float)index + 100.f, (float)index + 100.f, 5.f };

		Puck::Ptr puck = objects.emplace_back(std::make_shared<Puck>(name, modelName, pos));
		puck->setTypeActorPhysics(Engine::Physics::Type::CONVEX);
	}

	// Target
	{
		const string name = "Target";
		const string modelName = "TargetGreen";
		const vec3 pos{ 200.f, 200.f, 5.f };
		
		target = objects.emplace_back(std::make_shared<Target>(name, modelName, pos));
		target->setTypeActorPhysics(Engine::Physics::Type::CONVEX);
	}

	// Garbage
	for (int i = -15; i < 15; ++i) {
		for (int j = -15; j < 15; ++j) {
			const string name = "Garbage_"s + std::to_string(i) + std::to_string(j);
			const string modelName = "Box_01";
			const vec3 pos{ (float)i * 15.f, (float)j * 15.f, float(i+j)* 15.f };

			Object::Ptr object = objects.emplace_back(std::make_shared<Object>(name, modelName, pos));
			object->setTypeActorPhysics(Engine::Physics::Type::CONVEX);
		}
	}

	return true;
}

void MenuMap::action() {
	Map::action();

	if (_tact < 100) {
		++_tact;
		if (_tact == 100) {
			enableForce = true;
		}
	}
}
