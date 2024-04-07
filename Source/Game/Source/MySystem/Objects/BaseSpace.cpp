// ◦ Xyz ◦
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

	double mergeDistFactor = 1.f;
	using Indices = std::list<int>;
	std::unordered_map<int, Indices> mergeList;

	auto getForce = [&](size_t statIndex, size_t endIndex) {
		std::pair<int, Indices>* mergePair = nullptr;

		for (size_t index = statIndex; index <= endIndex; ++index) {
			BodyData::Data& data = _datas[index];

			float radius = _bodies[index]->Scale();

			double mass = data.mass;
			Math::Vector3& pos = data.pos;
			Math::Vector3& forceVec = data.force;
			forceVec.x = 0;
			forceVec.y = 0;
			forceVec.z = 0;

			for (size_t otherIndex = 0; otherIndex < sizeData; ++otherIndex) {
				BodyData::Data& otherBody = _datas[otherIndex];

				float otherRadius = _bodies[otherIndex]->Scale();

				if (&data == &otherBody) {
					continue;
				}

				Math::Vector3 gravityVec = otherBody.pos - pos;
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
	std::vector<Body::Ptr> newBodies;
	std::vector<BodyData::Data> newDatas;

	newBodies.reserve(2000);
	newDatas.reserve(2000);

	if (!mergeList.empty()) {
		for (auto& mergePair : mergeList) {
			if (_datas[mergePair.first].mass == 0 || !_bodies[mergePair.first]) {
				continue;
			}

			double sumMass = 0;
			Math::Vector3 sumPulse;
			Math::Vector3 sumForce;
			Math::Vector3 sumMassPos;

			//BodyData::Ptr newBody;
			Body::Ptr newBody;

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
				sumPulse += _bodies[index]->Velocity() * data.mass;

				if (!newBody) {
					double _mass_ = _bodies[index]->Mass();
					Math::Vector3 _velocity_ = _bodies[index]->Velocity();
					Math::Vector3 _pos_ = _bodies[index]->GetPos();
					std::string nameMode = _bodies[index]->HasModel() ? _bodies[index]->getModel().getName() : "BrownStone";

					newBody = Body::Ptr(new BodyData(nameMode));
					newBody->color = _bodies[index]->color;
					newBodies.emplace_back(newBody);
					static_cast<BodyData*>(newBody.get())->_dataPtr = &newDatas.emplace_back(sumMass, Math::Vector3(0, 0, 0), sumForce);
				}

				_bodies[index].reset();

				data.mass = 0;
			}

			static_cast<BodyData*>(newBody.get())->_dataPtr->force = sumForce;
			newBody->Mass() = sumMass;
			static_cast<BodyData*>(newBody.get())->_dataPtr->mass = sumMass;

			newBody->Velocity() = sumPulse / sumMass;
			newBody->SetPos(sumMassPos / sumMass);
		}
	}

	// ...
	float longBistanceFromStar = 150000.f;
	size_t needDataAssociation = std::numeric_limits<double>::min();
	std::vector<size_t> indRem;

	size_t size = _bodies.size();

	auto star = GetHeaviestBody();
	Math::Vector3 posStar = star ? star->GetPos() : Math::Vector3();

	for (size_t index = 0; index < size; ++index) {
		Body::Ptr& body = _bodies[index];
		if (!body) {
			continue;
		}

		static double minForce = std::numeric_limits<double>::min();
		if ((static_cast<BodyData*>(body.get())->_dataPtr->force.length() < minForce) && (star && (posStar - body->GetPos()).length() > longBistanceFromStar)) {
			indRem.emplace_back(index);
			++needDataAssociation;
			continue;
		}

		Math::Vector3 acceleration = static_cast<BodyData*>(body.get())->_dataPtr->force / body->Mass();
		Math::Vector3 newVelocity = acceleration * static_cast<double>(dt);

		body->Velocity() += newVelocity;

		static_cast<BodyData*>(body.get())->_dataPtr->pos += body->Velocity() * static_cast<double>(dt);
		body->SetPos(static_cast<BodyData*>(body.get())->_dataPtr->pos);

		body->Force() = static_cast<BodyData*>(body.get())->_dataPtr->force;
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
		Preparation();
	}

	//...
	if (!newBodies.empty()) {
		size_t newSize = newBodies.size();

		for (size_t index = 0; index < newSize; ++index) {
			auto& body = newBodies[index];

			Math::Vector3 acceleration = static_cast<BodyData*>(body.get())->_dataPtr->force / body->Mass();
			Math::Vector3 newVelocity = acceleration * static_cast<double>(dt);

			body->Velocity() += newVelocity;

			static_cast<BodyData*>(body.get())->_dataPtr->pos += body->Velocity() * static_cast<double>(dt);
			body->SetPos(static_cast<BodyData*>(body.get())->_dataPtr->pos);

			body->Force() = static_cast<BodyData*>(body.get())->_dataPtr->force;
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

		Preparation();
	}
}

void BaseSpace::Preparation() {
	_datas.clear();
	_datas.reserve(_bodies.size());

	std::sort(_bodies.begin(), _bodies.end(), [](const Body::Ptr& left, const Body::Ptr& right) {
		if (left && right) {
			return left->Mass() > right->Mass();
		}
		return left && !right;
	});

	for (Body::Ptr& body : _bodies) {
		if (!body) {
			continue;
		}

		static_cast<BodyData*>(body.get())->_dataPtr = &(_datas.emplace_back(body->Mass(), body->GetPos()));
		body->CalcScale();
	}

	size_t sizeInfo = 10;
	sizeInfo = sizeInfo > _bodies.size() ? _bodies.size() : 10;
	_heaviestInfo.clear();
	_heaviestInfo.reserve(sizeInfo);

	for (size_t index = 0; index < sizeInfo; ++index) {
		if (Body::Ptr& body = _bodies[index]) {
			_heaviestInfo.emplace_back(body, std::to_string(body->Mass()));
		}
	}
}

std::string BaseSpace::GetNameClass() {
	return Engine::GetClassName(this);
}
