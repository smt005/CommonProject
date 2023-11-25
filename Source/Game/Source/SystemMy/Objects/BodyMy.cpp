#include "BodyMy.h"
#include "Common/Help.h"
#include <glm/gtc/matrix_transform.hpp>

float BodyMy::G = 0.01f;
Map::Ptr BodyMy::system;
glm::vec3 BodyMy::centerSystem;
glm::vec3 BodyMy::centerMassSystem;
std::vector<Object*> BodyMy::removeObject;
BodyMy* BodyMy::centerBody = nullptr;

std::vector<BodyMy::ForceData> BodyMy::bodyes;

void BodyMy::ApplyForce(const double dt) {
	for (ForceData& data : bodyes) {
		glm::vec3 acceleration = data.forceVec3 / data.body->mass;
		glm::vec3 newVelocity = acceleration * (float)dt;

		data.body->_velocity += newVelocity;

		glm::vec3 pos = data.body->getPos();
		pos += data.body->_velocity * (float)dt;

		data.body->setPos(pos);
	}

	bodyes.clear();
}

void BodyMy::AddForceMy(const glm::vec3& forceVec) {
	bodyes.emplace_back(this, forceVec);
}

void BodyMy::action() {
	if (tag != 123) {
		return;
	}

	glm::vec3 gravityVector = GetVector();
	AddForceMy(gravityVector);
}

glm::vec3 BodyMy::GetVector() {
	glm::vec3 sumGravityVector = { 0.f, 0.f, 0.f };
	glm::vec3 pos = getPos();

	//...
	_points.push_back(pos);
	if (_points.size() > 100) {
		_points.pop_front();
	}
	const size_t countPoints = _points.size();
	float* pathPoints = new float[countPoints * 3];
	size_t index = 0;
	for (glm::vec3& p : _points) {
		pathPoints[index] = p.x;
		++index;
		pathPoints[index] = p.y;
		++index;
		pathPoints[index] = p.z;
		++index;
	}
	_path.set(pathPoints, countPoints);
	delete[] pathPoints;

	//...
	for (auto& objPtr : system->GetObjects()) {
		if (objPtr->tag != 123) {
			continue;
		}

		BodyMy* body = static_cast<BodyMy*>(objPtr.get());
		if (body == this) {
			continue;
		}

		glm::vec3 gravityVector = body->getPos() - pos;
		float dist = glm::length(gravityVector);
		gravityVector = glm::normalize(gravityVector);

		float force = G * (body->mass * mass) / (dist * dist);
		gravityVector *= force;

		sumGravityVector += gravityVector;
	}

	float sumGravity = glm::length(sumGravityVector);
	if (sumGravity < 0.0001f && getName() != "Sun") {
		removeObject.push_back(this);
	}

	float points[] = { pos.x, pos.y, pos.z, pos.x + sumGravityVector.x, pos.y + sumGravityVector.y, pos.z + sumGravityVector.z };
	_forceVector.set(points, 2);
	_forceVector.color = { 0.1f, 0.9f, 0.1f, 0.5f };

	return sumGravityVector;
}

Line& BodyMy::LineToCenter() {
	glm::vec3 pos = getPos();

	float points[] = { pos.x, pos.y, pos.z, centerSystem.x, centerSystem.y, centerSystem.z };
	_lineToCenter.set(points, 2);

	return _lineToCenter;
}

Line& BodyMy::LineToMassCenter() {
	glm::vec3 pos = getPos();

	float points[] = { pos.x, pos.y, pos.z, centerMassSystem.x, centerMassSystem.y, centerMassSystem.z };
	_lineToMassCenter.set(points, 2);

	return _lineToMassCenter;
}

void BodyMy::CalculateRelativelyLinePath() {
	if (!centerBody) {
		return;
	}

	if (_path.getCount() != centerBody->_path.getCount()) {
		return;
	}

	const size_t countPoints = _points.size();
	float* relativelyPoints = new float[countPoints * 3];
	size_t i = 0;

	auto itPoint = _points.begin();
	auto itEnd = _points.end();
	auto itCenterPoint = centerBody->_points.begin();

	std::vector<glm::vec3> rel;
	rel.reserve(countPoints);

	glm::vec3 centerPos = centerBody->getPos();

	while (itPoint != itEnd) {
		glm::vec3 d = *itPoint - *itCenterPoint;
		d += centerPos;
		rel.emplace_back(d);

		++itPoint;
		++itCenterPoint;
	}

	{
		float* pathPoints = new float[countPoints * 3];
		size_t index = 0;
		for (glm::vec3& p : rel) {
			pathPoints[index] = p.x;
			++index;
			pathPoints[index] = p.y;
			++index;
			pathPoints[index] = p.z;
			++index;
		}
		_relativelyPath.set(pathPoints, countPoints);
		delete[] pathPoints;
	}
}

// STATIC
glm::vec3 BodyMy::CenterSystem() {
	float count = 0.f;
	glm::vec3 centerPos = {0.f, 0.f, 0.f};

	for (auto& objPtr : system->GetObjects()) {
		if (objPtr->tag != 123) {
			continue;
		}

		centerPos += objPtr->getPos();
		count += 1.f;
	}

	centerSystem = centerPos / count;
	return centerSystem;
}

glm::vec3 BodyMy::CenterMassSystem() {
	float sumMass = 0.f;
	glm::vec3 sunPosMass = { 0.f, 0.f, 0.f };

	for (auto& objPtr : system->GetObjects()) {
		if (objPtr->tag != 123) {
			continue;
		}

		sunPosMass += objPtr->getPos() * objPtr->mass;
		sumMass += objPtr->mass;
	}

	centerMassSystem = sunPosMass / sumMass;
	return centerMassSystem;
}

void BodyMy::UpdateRalatovePos() {
	for (auto& objPtr : system->GetObjects()) {
		if (objPtr->tag != 123) {
			continue;
		}

		static_cast<BodyMy*>(objPtr.get())->CalculateRelativelyLinePath();
	}
}

void BodyMy::RemoveBody() {
	auto& objects = system->GetObjects();
	for (auto obj : removeObject) {
		auto it = std::find_if(objects.begin(), objects.end(), [obj](const auto& objPtr) {
			return objPtr.get() == obj;
		});

		if (it != objects.end()) {
			objects.erase(it);
		}
	}
}
