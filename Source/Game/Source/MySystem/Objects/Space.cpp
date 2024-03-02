#include "Space.h"

#include <thread>
#include <list>
#include <unordered_map>
#include "Common/Common.h"
#include "Common/Help.h"
#include "BodyData.h"

Space::Space(const std::string& name)
	: _name(name)
{}

Space::Space(Json::Value& valueData) {
	Load(valueData);
}

void Space::Save() {
	Json::Value jsonMap;
	jsonMap["name"] = _name;
	jsonMap["const_gravity"] = _constGravity;
	// TODO: 
	//jsonMap["class"] = Engine::GetClassName(this);

	std::string sc = GetNameClass();
	jsonMap["class"] = sc;

	/*if (_skyboxObject) {
		jsonMap["sky_box"] = _skyboxObject->getModel().getName();
	}*/

	for (Body::Ptr& body : Objects()) {
		Json::Value jsonObject;

		if (body->_name) {
			jsonObject["name"] = body->_name;
		}

		jsonObject["model"] = body->getModel().getName();
		jsonObject["mass"] = body->Mass();

		Math::Vector3 pos = body->GetPos();
		jsonObject["pos"][0] = pos.x;
		jsonObject["pos"][1] = pos.y;
		jsonObject["pos"][2] = pos.z;

		Math::Vector3 velocity = body->Velocity();
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

	std::string filePath = "Spaces/" + _name + ".json";
	help::saveJson(filePath, valueData);
}

bool Space::Load(Json::Value& valueData) {
	if (!valueData.isArray() || valueData.empty()) {
		return false;
	}

	Json::Value jsonParams = valueData[0]["params"].isObject() ? valueData[0]["params"] : Json::Value();
	if (!jsonParams.empty()) {
		for (auto const& key : jsonParams.getMemberNames()) {
			std::string value = jsonParams[key].asString();
			_params[key] = value;

			std::cout << key << ": " << value << std::endl;
		}
	}

	Json::Value jsonObjects = valueData[0]["objects"].isArray() ? valueData[0]["objects"] : Json::Value();
	if (jsonObjects.empty()) {
		return false;
	}

	_name = valueData[0]["name"].isString() ? valueData[0]["name"].asString() : std::string();
	_constGravity = valueData[0]["const_gravity"].isDouble() ? valueData[0]["const_gravity"].asDouble() : 0.01f;

	std::string skyBoxModel = valueData[0]["sky_box"].isString() ? valueData[0]["sky_box"].asString() : std::string();
	if (!skyBoxModel.empty()) {
		_skyboxObject = std::make_shared<Object>("SkyBox", skyBoxModel);
		if (!_skyboxObject->ValidModel()) {
			_skyboxObject.reset();
		}
	}
	
	processGPU = _params["PROCESS"] == "GPU" ? true : false;
	multithread = _params["MULTITHREAD"] == "true" ? true : false;
	tag = atoi(_params["TAG"].c_str());

	// Загрузка тел
	_bodies.clear();

	for (Json::Value& jsonObject : jsonObjects) {
		std::string model = jsonObject["model"].isString() ? jsonObject["model"].asString() : "";
		if (model.empty()) {
			continue;
		}

		std::string name = jsonObject["name"].isString() ? jsonObject["name"].asString() : "";
		double mass = jsonObject["mass"].isDouble() ? jsonObject["mass"].asDouble() : 1.f;

		Math::Vector3 pos(0.f, 0.f, 0.f);
		if (jsonObject["pos"].isArray()) {
			pos.x = jsonObject["pos"][0].asDouble();
			pos.y = jsonObject["pos"][1].asDouble();
			pos.z = jsonObject["pos"][2].asDouble();
		}

		Math::Vector3 vel(0.f, 0.f, 0.f);
		if (jsonObject["vel"].isArray()) {
			vel.x = jsonObject["vel"][0].asDouble();
			vel.y = jsonObject["vel"][1].asDouble();
			vel.z = jsonObject["vel"][2].asDouble();
		}

		BodyData* body = new BodyData(model);
		_bodies.emplace_back(body);

		if (!name.empty()) {
			body->SetName(name);
		}

		body->SetPos(pos);
		body->SetVelocity(vel);

		body->_mass = mass;
	}

	return true;
}

bool Space::Load() {
	std::string filePath = "Spaces/" + _name + ".json";
	Json::Value valueData;

	if (!help::loadJson(filePath, valueData)) {
		return false;
	}

	return Load(valueData);
}

Body::Ptr Space::GetHeaviestBody() {
	Body::Ptr heaviestBody;
	
	if (_bodies.empty()) {
		return heaviestBody;
	}
	
	for (Body::Ptr& body : _bodies) {
		if (body) {
			if (!heaviestBody) {
				heaviestBody = body;
			}
			else if (body->Mass() > heaviestBody->Mass()) {
				heaviestBody = body;
			}
		}
	}
	
	return heaviestBody;
}

Math::Vector3 Space::CenterMass() {
	double sumMass = 0;
	Math::Vector3 sunPosMass(0, 0, 0);

	for (Body::Ptr& body : _bodies) {
		sunPosMass += body->GetPos() * body->Mass();
		sumMass += body->Mass();
	}

	return sunPosMass / sumMass;
}

void Space::RemoveVelocity(bool toCenter) {
	if (_bodies.empty()) {
		return;
	}

	Math::Vector3 velocity;

	for (Body::Ptr& body : _bodies) {
		velocity += body->Velocity();
	}

	velocity /= static_cast<double>(_bodies.size());

	toCenter = toCenter && _focusBody;
	Math::Vector3 focusPos = toCenter ? _focusBody->GetPos() : Math::Vector3();

	for (Body::Ptr& body : _bodies) {
		body->Velocity() -= velocity;

		if (toCenter) {
			Math::Vector3 pos = _focusBody->GetPos();
			pos -= toCenter;
			_focusBody->SetPos(pos);
		}
	}
}

Body::Ptr Space::HitObject(const glm::mat4x4& matCamera) {
	for (Body::Ptr& body : _bodies) {
		if (body->hit(matCamera)) {
			return body;
		}
	}

	return Body::Ptr();
}

Body::Ptr Space::GetBody(const char* chName) {
	auto itBody = std::find_if(_bodies.begin(), _bodies.end(), [chName](const Body::Ptr& body) {
		return body ? (body->_name && chName && strcmp(body->_name, chName) == 0) : false;
	});
	return itBody != _bodies.end() ? *itBody : nullptr;
}

Body::Ptr& Space::Add(Body* body) {
	_bodies.emplace_back(body);
	return _bodies.back();
}

Body::Ptr& Space::Add(Body::Ptr& body) {
	_bodies.emplace_back(body);
	return body;
}

void Space::RemoveBody(Body::Ptr& body) {
	auto itRemove = std::find_if(_bodies.begin(), _bodies.end(), [&body](const Body::Ptr& itBody) { return itBody == body; });
	if (itRemove != _bodies.end()) {
		_bodies.erase(itRemove);
	}
}

std::pair<bool, Body&> Space::RefFocusBody() {
	auto it = std::find(_bodies.begin(), _bodies.end(), _focusBody);
	if (it != _bodies.end()) {
		Body& body = **it;
		return {true, body};
	}

	static Body defaultBody;
	return {false, defaultBody};
}

std::string Space::GetNameClass() {
	return Engine::GetClassName(this);
}
