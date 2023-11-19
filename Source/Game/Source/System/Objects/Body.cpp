#include "Body.h"
#include "Common/Help.h"
#include <glm/gtc/matrix_transform.hpp>

Map::Ptr Body::system;
glm::vec3 Body::centerSystem;
glm::vec3 Body::centerMassSystem;
std::vector<Object*> Body::removeObject;

void Body::action() {
	if (!hasPhysics()) {
		return;
	}

	glm::vec3 gravityVector = GetVector();
	addForce(gravityVector);
}

glm::vec3 Body::GetVector() {
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
	//_path.color = { 0.1f, 0.1f, 0.9f, 0.5f };

	//...
	for (auto& objPtr : system->GetObjects()) {
		if (!objPtr->hasPhysics() || objPtr->tag != 123) {
			continue;
		}

		Body* body = static_cast<Body*>(objPtr.get());
		if (body == this) {
			continue;
		}

		glm::vec3 gravityVector = body->getPos() - pos;
		float dist = glm::length(gravityVector);
		gravityVector = glm::normalize(gravityVector);

		float force = (body->mass * mass) / dist;
		gravityVector *= force;

		sumGravityVector += gravityVector;
	}

	float sumGravity = glm::length(sumGravityVector);
	if (sumGravity < 0.0001f && getName() != "Sun") {
		removeObject.push_back(this);
	}

	//float points[] = { pos.x, pos.y, pos.z, pos.x + sumGravityVector.x * 100.f, pos.y + sumGravityVector.y * 100.f, pos.z + sumGravityVector.z * 100.f };
	float points[] = { pos.x, pos.y, pos.z, pos.x + sumGravityVector.x, pos.y + sumGravityVector.y, pos.z + sumGravityVector.z };
	_forceVector.set(points, 2);
	_forceVector.color = { 0.1f, 0.9f, 0.1f, 0.5f };

	return sumGravityVector;
}

Line& Body::LineToCenter() {
	glm::vec3 pos = getPos();

	float points[] = { pos.x, pos.y, pos.z, centerSystem.x, centerSystem.y, centerSystem.z };
	_lineToCenter.set(points, 2);

	return _lineToCenter;
}

Line& Body::LineToMassCenter() {
	glm::vec3 pos = getPos();

	float points[] = { pos.x, pos.y, pos.z, centerMassSystem.x, centerMassSystem.y, centerMassSystem.z };
	_lineToMassCenter.set(points, 2);

	return _lineToMassCenter;
}

// STATIC
glm::vec3 Body::CenterSystem() {
	float count = 0.f;
	glm::vec3 centerPos = {0.f, 0.f, 0.f};

	for (auto& objPtr : system->GetObjects()) {
		if (!objPtr->hasPhysics() || objPtr->tag != 123) {
			continue;
		}

		centerPos += objPtr->getPos();
		count += 1.f;
	}

	centerSystem = centerPos / count;
	return centerSystem;
}

glm::vec3 Body::CenterMassSystem() {
	float sumMass = 0.f;
	glm::vec3 sunPosMass = { 0.f, 0.f, 0.f };

	for (auto& objPtr : system->GetObjects()) {
		if (!objPtr->hasPhysics() || objPtr->tag != 123) {
			continue;
		}

		sunPosMass += objPtr->getPos() * objPtr->mass;
		sumMass += objPtr->mass;
	}

	centerMassSystem = sunPosMass / sumMass;
	return centerMassSystem;
}

void Body::RemoveBody() {
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
