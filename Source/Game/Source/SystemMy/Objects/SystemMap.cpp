#include "SystemMap.h"

#if SYSTEM_MAP == 0

#include <thread>
#include "../../Engine/Source/Object/Model.h"
#include "../../Engine/Source/Common/Help.h"

using namespace S00;
Body::Body(const std::string& nameModel)
	: _model(Model::getByName(nameModel))
{}

Body::Body(const std::string& nameModel, const glm::vec3& pos, const glm::vec3& velocity, float mass, const std::string& name)
	: _mass(mass)
	, _velocity(velocity)
	, _model(Model::getByName(nameModel))
{
	SetPos(pos);
}

Body::~Body() {
	delete _name;
}

//... 
SystemMap::SystemMap(const std::string& name)
	: _name(name)
{}

SystemMap::~SystemMap() {
	for (Body* body : _bodies) {
		delete body;
	}
}

void SystemMap::Update(double dt, int countForceTime) {
	for (int index = 0; index < countForceTime; ++index) {
		Update(dt);
	}
}

void SystemMap::Update(double dt) {
	auto getForce = [&](size_t statIndex, size_t endIndex) {
		for (size_t index = statIndex; index < endIndex; ++index) {
			Body* body = _bodies[index];

			glm::vec3& forceVec = body->_force;

			for (Body* otherBody : _bodies) {
				if (body == otherBody) {
					continue;
				}

				glm::vec3 gravityVec = otherBody->GetPos() - body->GetPos();
				float dist = glm::length(gravityVec);
				gravityVec = glm::normalize(gravityVec);

				float force = _constGravity * (body->_mass * otherBody->_mass) / (dist * dist);
				gravityVec *= force;
				forceVec += gravityVec;
			}
		}
	};

	if (threadEnable) {
		unsigned int counThread = thread::hardware_concurrency();
		size_t size = _bodies.size();
		size_t dSize = _bodies.size() / counThread;
		dSize -= 1;

		vector<std::pair<size_t, size_t>> ranges;
		vector<std::thread> threads;
		threads.reserve(counThread);

		int statIndex = 0; size_t endIndex = statIndex + dSize;
		while(endIndex < size) {
			ranges.emplace_back(statIndex, endIndex);
			statIndex = ++endIndex; endIndex = statIndex + dSize;
		}

		for (auto& pair : ranges) {
			threads.emplace_back([&]() {
				getForce(pair.first, pair.second);
			});
		}

		for (thread& th : threads) {
			th.join();
		}
	} else {
		getForce(0, _bodies.size());
	}

	for (Body* body : _bodies) {
		glm::vec3 acceleration = body->_force / body->_mass;
		glm::vec3 newVelocity = acceleration * static_cast<float>(dt);

		body->_velocity += newVelocity;

		glm::vec3 pos = body->GetPos();
		pos += body->_velocity * static_cast<float>(dt);
		body->SetPos(pos);

		body->_force = { 0, 0, 0 };
	}

	if (dt > 0) {
		++time;
	} else {
		--time;
	}
}

void SystemMap::Save() {
	Json::Value jsonMap;
	jsonMap["name"] = _name;

	for (Body* body : Objects()) {
		Json::Value jsonObject;

		if (body->_name) {
			jsonObject["name"] = body->_name;
		}

		jsonObject["model"] = body->getModel().getName();
		jsonObject["mass"] = body->_mass;

		glm::vec3 pos = body->GetPos();
		jsonObject["pos"][0] = pos.x;
		jsonObject["pos"][1] = pos.y;
		jsonObject["pos"][2] = pos.z;

		glm::vec3 velocity = body->_velocity;
		jsonObject["vel"][0] = velocity.x;
		jsonObject["vel"][1] = velocity.y;
		jsonObject["vel"][2] = velocity.z;

		jsonMap["objects"].append(jsonObject);
	}

	/*if (auto camera = map.getCamera()) {
		camera->Save(jsonMap["Camera"]);
	}*/

	Json::Value valueData;
	valueData.append(jsonMap);

	std::string filePath = "SystemMaps/" + _name + ".json";
	help::saveJson(filePath, valueData);
}

bool SystemMap::Load() {
	std::string filePath = "SystemMaps/" + _name + ".json";
	Json::Value valueData;

	if (!help::loadJson(filePath, valueData) || !valueData.isArray() || valueData.empty()) {
		return false;
	}

	Json::Value& jsonObjects = valueData[0]["objects"].isArray() ? valueData[0]["objects"] : Json::Value();

	if (jsonObjects.empty()) {
		return false;
	}

	std::string nameMap = valueData[0]["name"].isString() ? valueData[0]["name"].asString() : "EMPTY_NAME";

	for (Json::Value& jsonObject : jsonObjects) {
		std::string model = jsonObject["model"].isString() ? jsonObject["model"].asString() : "";
		if (model.empty()) {
			continue;
		}

		std::string name = jsonObject["name"].isString() ? jsonObject["name"].asString() : "";
		float mass = jsonObject["mass"].isDouble() ? jsonObject["mass"].asDouble() : 1.f;

		glm::vec3 pos(0.f, 0.f, 0.f);
		if (jsonObject["pos"].isArray()) {
			pos.x = jsonObject["pos"][0].asDouble();
			pos.y = jsonObject["pos"][1].asDouble();
			pos.z = jsonObject["pos"][2].asDouble();
		}

		glm::vec3 vel(0.f, 0.f, 0.f);
		if (jsonObject["vel"].isArray()) {
			vel.x = jsonObject["vel"][0].asDouble();
			vel.y = jsonObject["vel"][1].asDouble();
			vel.z = jsonObject["vel"][2].asDouble();
		}

		Body& body = Add(model);

		if (!name.empty()) {
			body.SetName(name);
		}

		body.SetPos(pos);
		body.SetVelocity(vel);

		body._mass = mass;
	}

	return true;
}

glm::vec3 SystemMap::CenterMass() {
	float sumMass = 0;
	glm::vec3 sunPosMass(0, 0, 0);

	for (Body* body : _bodies) {
		sunPosMass += body->GetPos() * body->_mass;
		sumMass += body->_mass;
	}

	return sunPosMass / sumMass;
}

Body* SystemMap::GetBody(const char* chName) {
	auto itBody = std::find_if(_bodies.begin(), _bodies.end(), [chName](const Body* body) {
		return body->_name && chName && strcmp(body->_name, chName) == 0;
	});
	return itBody != _bodies.end() ? *itBody : nullptr;
}

#endif
