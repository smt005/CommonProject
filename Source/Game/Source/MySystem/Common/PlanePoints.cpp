// ◦ Xyz ◦

#include "PlanePoints.h"
#include <cmath>
//#include <glm/mat3x3.hpp>
#include <Draw2/Draw2.h>
#include <Draw2/Shader/ShaderLine.h>

void PlanePoints::Init(float space, float offset)
{
	_space = space;
	_offset = offset;

	size_t countPoints = std::pow(((_space * 2) / _offset + 1), 2);
	_points.reserve(countPoints);
	float z = 0.f;

	for (float x = -_space; x <= _space; x += _offset) {
		for (float y = -_space; y <= _space; y += _offset) {
			_points.emplace_back(x, y, z);
		}
	}

	if (!ShaderGravityPoint::Instance().Inited()) {
		ShaderGravityPoint::Instance().Init("GravityPoint.vert", "GravityPoint.frag");
	}
}

void PlanePoints::Update(std::vector<Body::Ptr>& objects)
{
	float constGravity = -100.f;
	float mass = 100.f;

	for (Math::Vector3& point : _points) {
		point.z = 0.f;
	}

	for (const Body::Ptr& bodyPtr : objects) {
		Math::Vector3 pos = bodyPtr->GetPos();

		for (Math::Vector3& point : _points) {
			Math::Vector3 dV = pos - point; // TODO: не работает с const
			float dist = std::abs(dV.length());
			double force = constGravity * (mass * bodyPtr) / (dist * dist);

			point.z += force;
		}
	}
}

void PlanePoints::Draw()
{
	ShaderGravityPoint::Instance().Use();

	static float sizePoint = 2.f;
	Draw2::SetPointSize(sizePoint);

	static float color4[] = { 0.f, 0.f, 0.0f, 0.25f };
	Draw2::SetColorClass<ShaderGravityPoint>(color4);

	Draw2::drawPoints((float*)_points.data(), _points.size());
}
