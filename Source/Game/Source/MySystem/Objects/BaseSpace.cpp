#include "BaseSpace.h"
#include <stdio.h>
#include <unordered_map>
#include <algorithm>
#include "Common/Common.h"
#include <Core.h>

void BaseSpace::Update(double dt) {
	for (size_t i = 0; i < countOfIteration; ++i) {
		Update();
	}
}

void BaseSpace::Update() {
	double dt = (double)deltaTime;

	size_t sizeData = _datas.size();
	if (sizeData <= 1) {
		return;
	}

	double mergeDistFactor = 10.f;
	using Indices = std::list<int>;
	std::unordered_map<int, Indices> mergeList;

	auto getForce = [&](size_t statIndex, size_t endIndex) {
		std::pair<int, Indices>* mergePair = nullptr;

		for (size_t index = statIndex; index <= endIndex; ++index) {
			BodyData::Data& data = _datas[index];

			float radius = _bodies[index]->_scale;

			double mass = data.mass;
			Math::Vector3d& pos = data.pos;
			Math::Vector3d& forceVec = data.force;
			forceVec.x = 0;
			forceVec.y = 0;
			forceVec.z = 0;

			for (size_t otherIndex = 0; otherIndex < sizeData; ++otherIndex) {
				BodyData::Data& otherBody = _datas[otherIndex];

				float otherRadius = _bodies[otherIndex]->_scale;

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
				float mergeDist = (radius + otherRadius) * mergeDistFactor;
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


	getForce(0, _bodies.size() - 1);

	//...
	std::vector<BodyData::Ptr> newBodies;
	std::vector<BodyData::Data> newDatas;

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
			Math::Vector3d sumMassPos;

			BodyData::Ptr newBody;

			for (auto& index : mergePair.second) {
				if (!_bodies[index]) {
					continue;
				}

				BodyData::Data& data = _datas[index];
				if (data.mass == 0) {
					continue;
				}

				sumForce += data.force;
				sumMassPos += (data.pos * data.mass);
				sumMass += data.mass;
				sumPulse += _bodies[index]->_velocity * data.mass;

				if (!newBody) {
					double _mass_ = _bodies[index]->_mass;
					Math::Vector3d _velocity_ = _bodies[index]->_velocity;
					Math::Vector3d _pos_ = _bodies[index]->GetPos();
					std::string nameMode = _bodies[index]->GetModel() ? _bodies[index]->getModel().getName() : "BrownStone";

					newBody = newBodies.emplace_back(new BodyData(nameMode));
					newBody->_dataPtr = &newDatas.emplace_back(sumMass, Math::Vector3d(0, 0, 0), sumForce);
				}

				_bodies[index].reset();

				data.mass = 0;
			}

			newBody->_dataPtr->force = sumForce;
			newBody->_mass = sumMass;
			newBody->_dataPtr->mass = sumMass;

			newBody->_velocity = sumPulse / sumMass;
			newBody->SetPos(sumMassPos / sumMass);
		}
	}

	// ...
	float longÂistanceFromStar = 150000.f;
	size_t needDataAssociation = std::numeric_limits<double>::min();
	std::vector<size_t> indRem;

	size_t size = _bodies.size();

	BodyData::Ptr star = GetHeaviestBody();
	Math::Vector3d posStar = star ? star->GetPos() : Math::Vector3d();

	for (size_t index = 0; index < size; ++index) {
		BodyData::Ptr& body = _bodies[index];
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

		body->force = body->_dataPtr->force.length();
	}

	if (needDataAssociation > 0) {
		std::vector<BodyData::Ptr> bodies;
		bodies.reserve(_bodies.size() - needDataAssociation);

		for (BodyData::Ptr& body : _bodies) {
			if (body) {
				bodies.emplace_back(body);
			}
		}

		std::swap(bodies, _bodies);
		Preparation();
	}

	//...

	if (!newBodies.empty()) {
		size_t newSize = newBodies.size();

		for (size_t index = 0; index < newSize; ++index) {
			BodyData::Ptr& body = newBodies[index];

			Math::Vector3d acceleration = body->_dataPtr->force / body->_mass;
			Math::Vector3d newVelocity = acceleration * static_cast<double>(dt);

			body->_velocity += newVelocity;

			body->_dataPtr->pos += body->_velocity * static_cast<double>(dt);
			body->SetPos(body->_dataPtr->pos);

			body->force = body->_dataPtr->force.length();
		}

		std::vector<BodyData::Ptr> bodies;
		bodies.reserve(_bodies.size());

		for (BodyData::Ptr body : _bodies) {
			if (body) {
				bodies.emplace_back(body);
			}
		}

		for (BodyData::Ptr bodyFromNew : newBodies) {
			if (bodyFromNew) {
				bodies.emplace_back(bodyFromNew);
			}
		}

		std::swap(bodies, _bodies);

		Preparation();
	}
}

void BaseSpace::Preparation() {
	double lastTime = Engine::Core::currentTime();

	_datas.clear();
	_datas.reserve(_bodies.size());

	std::sort(_bodies.begin(), _bodies.end(), [](const BodyData::Ptr& left, const BodyData::Ptr& right) {
		if (left && right) {
			return left->_mass > right->_mass;
		}
	return left && !right;
		});

	for (BodyData::Ptr& body : _bodies) {
		if (!body) {
			continue;
		}

		body->_dataPtr = &(_datas.emplace_back(body->_mass, body->GetPos()));
		body->Scale();
	}

	size_t sizeInfo = 10;
	sizeInfo = sizeInfo > _bodies.size() ? _bodies.size() : 10;
	_heaviestInfo.clear();
	_heaviestInfo.reserve(sizeInfo);

	for (size_t index = 0; index < sizeInfo; ++index) {
		if (BodyData::Ptr& body = _bodies[index]) {
			_heaviestInfo.emplace_back(body, std::to_string(body->_mass));
		}
	}

	lastTime = Engine::Core::currentTime() - lastTime;
	printf("BaseSpace::Preparation: %f size: %i\n", lastTime, _bodies.size());
}

std::string BaseSpace::GetNameClass() {
	return Engine::GetClassName(this);
}
