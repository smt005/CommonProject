#include "SaveManager.h"
#include <memory>
#include <string>
#include <glm/vec3.hpp>

#include "../../Engine/Source/Common/Help.h"
#include "../../Engine/Source/Object/Map.h"
#include "../../Engine/Source/Object/Object.h"
#include "../../Engine/Source/Object/Model.h"
#include "../../Engine/Source/Object/Identify.h"
#include "../../Engine/Source/Draw/Camera/CameraControlOutside.h"
#include "Objects/BodyMy.h"

namespace {
	std::string filePath = "Systems.json";
}

void SaveManager::GetMap() {
	Load();

	if (!(BodyMy::system = Map::GetFirstCurrentMapPtr())) {
		BodyMy::system = Map::AddCurrentMap(Map::Ptr(new Map("GenerateMap")));
	}

	BodyMy* sun = nullptr;
	if (Object::Ptr objectPtr = BodyMy::system->getObjectPtrByName("Sun")) {
		sun = dynamic_cast<BodyMy*>(objectPtr.get());
	}

	if (!sun) {
		Object::Ptr sunPtr = std::make_shared<BodyMy>("Sun", "OrangeStar", glm::vec3(0.f, 0.f, 0.f));
		sun = static_cast<BodyMy*>(sunPtr.get());
		sun->setMass(1000000.f);
		BodyMy::system->addObject(sunPtr);
	}

	BodyMy::_suns.emplace_back(sun);
}

void SaveManager::Save() {
	Map& map = Map::GetFirstCurrentMap();

	Json::Value jsonMap;
	jsonMap["name"] = map.getName();
	
	for (Object::Ptr objectPtr : map.GetObjects()) {
		if (objectPtr->tag != 123) {
			continue;
		}

		BodyMy& object = *(static_cast<BodyMy*>(objectPtr.get()));

		Json::Value jsonObject;

		jsonObject["name"] = object.getName();
		jsonObject["model"] = object.getModel().getName();
		jsonObject["mass"] = object.mass;

		glm::vec3 pos = object.getPos();
		jsonObject["pos"][0] = pos.x;
		jsonObject["pos"][1] = pos.y;
		jsonObject["pos"][2] = pos.z;

		glm::vec3 velocity = object._velocity;
		jsonObject["vel"][0] = velocity.x;
		jsonObject["vel"][1] = velocity.y;
		jsonObject["vel"][2] = velocity.z;

		jsonMap["objects"].append(jsonObject);
	}

	if (auto camera = map.getCamera()) {
		camera->Save(jsonMap["Camera"]);
	}

	Json::Value valueData;
	valueData.append(jsonMap);
	help::saveJson(filePath, valueData);
}

bool SaveManager::Load() {
	Json::Value valueData;
	if (!help::loadJson(filePath, valueData) || !valueData.isArray() || valueData.empty()) {
		return false;
	}

	Json::Value& jsonObjects = valueData[0]["objects"].isArray() ? valueData[0]["objects"] : Json::Value();

	if (jsonObjects.empty()) {
		return false;
	}

	std::string nameMap = valueData[0]["name"].isString() ? valueData[0]["name"].asString() : "EMPTY_NAME";

	Map::DataClass::clear();
	Map::Ptr mainMap(new Map(nameMap));
	Map::AddCurrentMap(Map::add(mainMap));

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

		BodyMy* body = new BodyMy(name, model, pos);
		body->setMass(mass);
		body->_velocity = vel;

		mainMap->addObject(body);
	}

	return true;
}

void SaveManager::Reload() {
	Map::DataClass::clear();
	BodyMy::system.reset();
	BodyMy::_suns.clear();
	BodyMy::_curentSunn = 0;

	GetMap();
}
