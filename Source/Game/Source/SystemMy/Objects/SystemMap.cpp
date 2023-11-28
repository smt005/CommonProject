#include "SystemMap.h"
#include "BodyMy.h"
#include "../../Engine/Source/Object/Model.h"
#include "../../Engine/Source/Common/Help.h"

Body::Body(const std::string& nameModel) {
	_model = Model::getByName(nameModel);
}

Body::~Body() {
	delete _name;
}

SystemMap::~SystemMap() {
	for (Body* body : _bodies) {
		delete body;
	}
}

void SystemMap::Update(double dt) {
	for (Body* body : _bodies) {
		Vector3& forceVec = body->_force;

		for (Body* otherBody : _bodies) {
			if (body == otherBody) {
				continue;
			}

			Vector3 gravityVec = otherBody->GetPos() - body->GetPos();
			ValueT dist = glm::length(gravityVec);
			gravityVec = glm::normalize(gravityVec);

			ValueT force = _constGravity * (body->_mass * otherBody->_mass) / (dist * dist);
			gravityVec *= force;

			forceVec += gravityVec;
		}
	}

	for (Body* body : _bodies) {
		Vector3 acceleration = body->_force / body->_mass;
		Vector3 newVelocity = acceleration * static_cast<ValueT>(dt);

		body->_velocity += newVelocity;

		Vector3 pos = body->GetPos();
		pos += body->_velocity * static_cast<ValueT>(dt);
		body->SetPos(pos);

		body->_force = { 0, 0, 0 };
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

		Vector3 pos(0.f, 0.f, 0.f);
		if (jsonObject["pos"].isArray()) {
			pos.x = jsonObject["pos"][0].asDouble();
			pos.y = jsonObject["pos"][1].asDouble();
			pos.z = jsonObject["pos"][2].asDouble();
		}

		Vector3 vel(0.f, 0.f, 0.f);
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
