#include "SystemMapStackArr.h"
#if SYSTEM_MAP == 2

#include <thread>
#include "../Objects/SystemClass.h"
#include "../../Engine/Source/Object/Model.h"
#include "../../Engine/Source/Common/Help.h"

using namespace SARR;

SystemStackData* SystemStackData::dataPtr = nullptr;

Body::Body(const std::string& nameModel)
	: _model(Model::getByName(nameModel))
{}

Body::Body(const std::string& nameModel, const Vector3& pos, const Vector3& velocity, ValueT mass, const std::string& name)
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
	size_t sizeData = SystemStackData::dataPtr->size;
	if (sizeData <= 1) {
		return;
	}
	
	Body::Data* datas = SystemStackData::dataPtr->bodies;

	auto getForce = [&](size_t statIndex, size_t endIndex) {
		for (size_t index = statIndex; index <= endIndex; ++index) {
			Body::Data& data = datas[index];

			ValueT mass = data.mass;
			Vector3& pos = data.pos;
			Vector3& forceVec = data.force;
			forceVec.x = 0;
			forceVec.y = 0;
			forceVec.z = 0;

			for (size_t otherIndex = 0; otherIndex < sizeData; ++otherIndex) {
				Body::Data& otherBody = datas[otherIndex];

				if (&data == &otherBody) {
					continue;
				}

				Vector3 gravityVec = otherBody.pos - pos;
				ValueT dist = glm::length(gravityVec);
				gravityVec = glm::normalize(gravityVec);

				ValueT force = _constGravity * (mass * otherBody.mass) / (dist * dist);
				gravityVec *= force;
				forceVec += gravityVec;
			}
		}
	};

	unsigned int counThread = static_cast<double>(thread::hardware_concurrency());
	int lastIndex = _bodies.size() - 1;

	if (threadEnable && ((lastIndex * 2) > counThread)) {
		double counThreadD = static_cast<double>(counThread);
		
		double lastIndexD = static_cast<double>(lastIndex);
		float dSizeD = lastIndexD / counThreadD;
		int dSize = static_cast<int>(round(dSizeD));
		dSize = dSize == 0 ? 1 : dSize;

		vector<std::pair<size_t, size_t>> ranges;
		vector<std::thread> threads;
		threads.reserve(counThread);

		int statIndex = 0; size_t endIndex = statIndex + dSize;
		while(statIndex < lastIndex) {
			ranges.emplace_back(statIndex, endIndex);
			statIndex = ++endIndex; endIndex = statIndex + dSize;
		}

		ranges.back().second = lastIndex;
		for (auto& pair : ranges) {
			threads.emplace_back([&]() {
				getForce(pair.first, pair.second);
			});
		}

		for (thread& th : threads) {
			th.join();
		}
	} else {
		getForce(0, _bodies.size() - 1);
	}

	for (Body* body : _bodies) {
		if (body->_dataPtr == nullptr) {
			continue;
		}

		Vector3 acceleration = body->_dataPtr->force / body->_mass;
		Vector3 newVelocity = acceleration * static_cast<ValueT>(dt);

		body->_velocity += newVelocity;

		body->_dataPtr->pos += body->_velocity * static_cast<ValueT>(dt);
		body->SetPos(body->_dataPtr->pos);
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
	_bodies.clear();

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

		Body* body = new Body(model);
		_bodies.emplace_back(body);

		if (!name.empty()) {
			body->SetName(name);
		}

		body->SetPos(pos);
		body->SetVelocity(vel);

		body->_mass = mass;
	}

	DataAssociation();

	return true;
}

void SystemMap::DataAssociation() {
	size_t size = _bodies.size();
	Body::Data* bodies = SystemStackData::dataPtr->bodies;

	for (size_t index = 0; index < size; ++index) {
		Body* body = _bodies[index];
		Body::Data& data = bodies[index];

		data.mass = body->_mass;
		data.pos = body->GetPos();

		body->_dataPtr = &data;
	}

	SystemStackData::dataPtr->size = size;
}

Vector3 SystemMap::CenterMass() {
	ValueT sumMass = 0;
	Vector3 sunPosMass(0, 0, 0);

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
