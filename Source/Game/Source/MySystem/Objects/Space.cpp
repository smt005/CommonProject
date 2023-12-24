#include "Space.h"

#include <thread>
#include <list>
#include <unordered_map>
#include "../../Engine/Source/Common/Help.h"

Space::Space(const std::string& name)
	: _name(name)
{}

Space::~Space() {
}

void Space::Update(double dt, int countForceTime) {
	for (int index = 0; index < countForceTime; ++index) {
		Update(dt);
	}
}

void Space::Update(double dt) {
	size_t sizeData = _datas.size();
	if (sizeData <= 1) {
		return;
	}

	double mergeDist = 100.f;
	using Indices = std::list<int>;
	std::unordered_map<int, Indices> mergeList;

	auto getForce = [&](size_t statIndex, size_t endIndex) {
		std::pair<int, Indices>* mergePair = nullptr;

		for (size_t index = statIndex; index <= endIndex; ++index) {
			Body::Data& data = _datas[index];

			double mass = data.mass;
			Math::Vector3d& pos = data.pos;
			Math::Vector3d& forceVec = data.force;
			forceVec.x = 0;
			forceVec.y = 0;
			forceVec.z = 0;

			for (size_t otherIndex = 0; otherIndex < sizeData; ++otherIndex) {
				Body::Data& otherBody = _datas[otherIndex];

				if (&data == &otherBody) {
					continue;
				}

				Math::Vector3d gravityVec = otherBody.pos - pos;
				double dist = Math::length(gravityVec);
				gravityVec = Math::normalize(gravityVec);

				double force = _constGravity * (mass * otherBody.mass) / (dist * dist);
				gravityVec *= force;
				forceVec += gravityVec;

				//...
				if (dist < mergeDist) {
					if (!mergePair) {
						mergePair = new std::pair<int, Indices>(index, Indices());
						mergePair->second.push_back(index);
					}

					mergePair->second.push_back(otherIndex);
				}
			}

			if (mergePair) {
				// TEMP_ 
				mergeList.emplace(std::move(*mergePair));
				delete mergePair;
				mergePair = nullptr;
			}
		}
	};

	unsigned int counThread = static_cast<double>(thread::hardware_concurrency());
	int lastIndex = _bodies.size() - 1;

	/*if (threadEnable && ((lastIndex * 2) > counThread)) {
		double counThreadD = static_cast<double>(counThread);
		
		double lastIndexD = static_cast<double>(lastIndex);
		double dSizeD = lastIndexD / counThreadD;
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
	} else*/
	{
		getForce(0, _bodies.size() - 1);
	}

	//...
	
	std::vector<Body::Ptr> newBodies;
	std::vector<Body::Data> newDatas;

	newBodies.reserve(2000);
	newDatas.reserve(2000);

	if (!mergeList.empty()) {
		for (auto& mergePair : mergeList) {
			if (_datas[mergePair.first].mass == 0 || !_bodies[mergePair.first]) {
				continue;
			}

			double sumMass = 0;
			Math::Vector3d sumPulse;
			Math::Vector3d sumForce;
			Math::Vector3d sumPos;
			double countPos = 0;

			Body::Ptr newBody;

			for (auto& index : mergePair.second) {
				if (!_bodies[index]) {
					continue;
				}

				Body::Data& data = _datas[index];
				if (data.mass == 0) {
					continue;
				}

				sumForce += data.force;
				sumPos += data.pos;
				sumMass += data.mass;
				sumPulse += _bodies[index]->_velocity * data.mass;
			
				if (!newBody) {
					double _mass_ = _bodies[index]->_mass;
					Math::Vector3d _velocity_ = _bodies[index]->_velocity;
					Math::Vector3d _pos_ = _bodies[index]->GetPos();
					std::string nameMode = _bodies[index]->HetModel() ? _bodies[index]->getModel().getName() : "BrownStone";

					newBody = newBodies.emplace_back(new Body(nameMode));
					//newBody = std::make_shared<Body>(new Body(nameMode));
					//newBodies.emplace_back(newBody);
					newBody->_dataPtr = &newDatas.emplace_back(sumMass, sumPos, sumForce);
				}

				_bodies[index].reset();

				data.mass = 0;
				countPos += 1;
			}

			sumPos /= countPos;

			newBody->_dataPtr->force = sumForce;
			newBody->_mass = sumMass;
			newBody->_dataPtr->mass = sumMass;
			
			newBody->_velocity = sumPulse / sumMass;
		}
	}

	// ...
	float longÂistanceFromStar = 150000.f;
	size_t needDataAssociation = std::numeric_limits<double>::min();
	std::vector<size_t> indRem;

	size_t size = _bodies.size();
	
	Body::Ptr star = GetHeaviestBody();
	Math::Vector3d posStar = star ? star->GetPos() : Math::Vector3d();

	for (size_t index = 0; index < size; ++index) {
		Body::Ptr& body = _bodies[index];
		if (!body) {
			continue;
		}

		static double minForce = std::numeric_limits<double>::min();
		if ((body->_dataPtr->force.length() < minForce) && (star && (posStar - body->GetPos()).length() > longÂistanceFromStar)) {
			indRem.emplace_back(index);
			++needDataAssociation;
			continue;
		}

		Math::Vector3d acceleration = body->_dataPtr->force / body->_mass;
		Math::Vector3d newVelocity = acceleration * static_cast<double>(dt);

		body->_velocity += newVelocity;

		body->_dataPtr->pos += body->_velocity * static_cast<double>(dt);
		body->SetPos(body->_dataPtr->pos);
	}

	if (needDataAssociation > 0) {
		std::vector<Body::Ptr> bodies;
		bodies.reserve(_bodies.size() - needDataAssociation);

		for (Body::Ptr& body : _bodies) {
			if (body) {
				bodies.emplace_back(body);
			}
		}

		std::swap(bodies, _bodies);
		DataAssociation();
	}

	//...

	if (!newBodies.empty()) {
		size_t newSize = newBodies.size();

		for (size_t index = 0; index < newSize; ++index) {
			Body::Ptr& body = newBodies[index];

			Math::Vector3d acceleration = body->_dataPtr->force / body->_mass;
			Math::Vector3d newVelocity = acceleration * static_cast<double>(dt);

			body->_velocity += newVelocity;

			body->_dataPtr->pos += body->_velocity * static_cast<double>(dt);
			body->SetPos(body->_dataPtr->pos);
		}

		std::vector<Body::Ptr> bodies;
		bodies.reserve(_bodies.size());

		for (Body::Ptr body : _bodies) {
			if (body) {
				bodies.emplace_back(body);
			}
		}

		for (Body::Ptr bodyFromNew : newBodies) {
			if (bodyFromNew) {
				bodies.emplace_back(bodyFromNew);
			}
		}

		std::swap(bodies, _bodies);

		DataAssociation();
	}

	//...
	if (dt > 0) {
		++time;
	} else {
		--time;
	}
}

void Space::Save() {
	Json::Value jsonMap;
	jsonMap["name"] = _name;

	for (Body::Ptr& body : Objects()) {
		Json::Value jsonObject;

		if (body->_name) {
			jsonObject["name"] = body->_name;
		}

		jsonObject["model"] = body->getModel().getName();
		jsonObject["mass"] = body->_mass;

		Math::Vector3d pos = body->GetPos();
		jsonObject["pos"][0] = pos.x;
		jsonObject["pos"][1] = pos.y;
		jsonObject["pos"][2] = pos.z;

		Math::Vector3d velocity = body->_velocity;
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

bool Space::Load() {
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
		double mass = jsonObject["mass"].isDouble() ? jsonObject["mass"].asDouble() : 1.f;

		Math::Vector3d pos(0.f, 0.f, 0.f);
		if (jsonObject["pos"].isArray()) {
			pos.x = jsonObject["pos"][0].asDouble();
			pos.y = jsonObject["pos"][1].asDouble();
			pos.z = jsonObject["pos"][2].asDouble();
		}

		Math::Vector3d vel(0.f, 0.f, 0.f);
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

void Space::DataAssociation() {
	_datas.clear();
	_datas.reserve(_bodies.size());

	std::sort(_bodies.begin(), _bodies.end(), [](const Body::Ptr& left, const Body::Ptr& right) {
		if (left && right) {
			return left->_mass > right->_mass;
		}
		return left && !right;
	});

	for (Body::Ptr& body : _bodies) {
		if (!body) {
			continue;
		}

		body->_dataPtr = &(_datas.emplace_back(body->_mass, body->GetPos()));
	}

	size_t sizeInfo = 10;
	sizeInfo = sizeInfo > _bodies.size() ? _bodies.size() : 10;
	_heaviestInfo.clear();
	_heaviestInfo.reserve(sizeInfo);

	for (size_t index = 0; index < sizeInfo; ++index) {
		if (Body::Ptr& body = _bodies[index]) {
			_heaviestInfo.emplace_back(body, std::to_string(body->_mass));
		}
	}
}

Body::Ptr Space::GetHeaviestBody(bool setAsStar) {
	if (_bodies.empty()) {
		return nullptr;
	}

	Body::Ptr heaviestBody;

	for (Body::Ptr& body : _bodies) {
		if (body) {
			if (!heaviestBody) {
				heaviestBody = body;
			} else if (body->_mass > heaviestBody->_mass) {
				heaviestBody = body;
			}
		}
	}

	if (heaviestBody && setAsStar) {
		heaviestBody->_model = Model::getByName("OrangeStar");
	}

	return heaviestBody;
}


Math::Vector3d Space::CenterMass() {
	double sumMass = 0;
	Math::Vector3d sunPosMass(0, 0, 0);

	for (Body::Ptr& body : _bodies) {
		sunPosMass += body->GetPos() * body->_mass;
		sumMass += body->_mass;
	}

	return sunPosMass / sumMass;
}

void Space::RemoveBody(Body::Ptr& body) {
	auto itRemove = std::find_if(_bodies.begin(), _bodies.end(), [&body](const Body::Ptr& itBody) { return itBody == body; });
	if (itRemove != _bodies.end()) {
		_bodies.erase(itRemove);
		DataAssociation();
	}
}

void Space::RemoveVelocity(bool toCenter) {
	if (_bodies.empty()) {
		return;
	}

	Math::Vector3d velocity;

	for (Body::Ptr& body : _bodies) {
		velocity += body->_velocity;
	}

	velocity /= static_cast<double>(_bodies.size());

	toCenter = toCenter && _focusBody;
	Math::Vector3d focusPos = toCenter ? _focusBody->GetPos() : Math::Vector3d();

	for (Body::Ptr& body : _bodies) {
		body->_velocity -= velocity;

		if (toCenter) {
			Math::Vector3d pos = _focusBody->GetPos();
			pos -= toCenter;
			_focusBody->SetPos(pos);
		}
	}
}

Body::Ptr Space::GetBody(const char* chName) {
	auto itBody = std::find_if(_bodies.begin(), _bodies.end(), [chName](const Body::Ptr& body) {
		return body ? (body->_name && chName && strcmp(body->_name, chName) == 0) : false;
	});
	return itBody != _bodies.end() ? *itBody : nullptr;
}

Body::Ptr Space::HitObject(const glm::mat4x4& matCamera) {
	for (Body::Ptr& body : _bodies) {
		if (body->hit(matCamera)) {
			return body;
		}
	}

	return Body::Ptr();
}