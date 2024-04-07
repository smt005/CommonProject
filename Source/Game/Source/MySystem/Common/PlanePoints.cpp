// ◦ Xyz ◦

#include "PlanePoints.h"
#include <cmath>
#include <glm/mat4x4.hpp>
#include <Draw2/Draw2.h>
#include <Draw2/Shader/ShaderLine.h>
#include <Common/Help.h>

void PlanePoints::Init(float spaceRange, float offset)
{
	_spaceRange = spaceRange;
	_offset = offset;

	float z = 0.f;

	for (float x = -_spaceRange; x <= _spaceRange; x += _offset) {
		for (float y = -_spaceRange; y <= _spaceRange; y += _offset) {
			Math::Vector3 vec(x, y, 0.f);
			if (vec.length() < _spaceRange) {
				_points.emplace_back(vec);
			}
		}
	}


	float x = -_spaceRange;
	float y = -_spaceRange;
	float direct = _offset;

	// X
	{
		while (true) {
			if (x > _spaceRange) {
				x = _spaceRange;
				y += _offset;
				direct = -_offset;
			}
			if (x < -_spaceRange) {
				x = -_spaceRange;
				y += _offset;
				direct = _offset;
			}
			if (y > _spaceRange) {
				y = _spaceRange;
				break;
			}

			_line.emplace_back(x, y, 0.f);

			x += direct;
		}
	}

	// Y
	{
		_line.reserve(_line.size() * 2);
		direct = -_offset;

		while (true) {
			if (y > _spaceRange) {
				y = _spaceRange;
				x -= _offset;
				direct = -_offset;
			}
			if (y < -_spaceRange) {
				y = -_spaceRange;
				x -= _offset;
				direct = _offset;
			}
			if (x < -_spaceRange) {
				break;
			}

			_line.emplace_back(x, y, 0.f);

			y += direct;
		}
	}

	if (!ShaderGravityPoint::Instance().Inited()) {
		ShaderGravityPoint::Instance().Init("GravityPoint.vert", "GravityPoint.frag");
	}
}

void PlanePoints::Update(std::vector<Body::Ptr>& objects)
{
	for (Math::Vector3& point : _points) {
		point.z = 0.f;
	}
	for (Math::Vector3& point : _line) {
		point.z = 0.f;
	}

	for (const Body::Ptr& bodyPtr : objects) {
		Math::Vector3 pos = bodyPtr->GetPos();

		for (Math::Vector3& point : _points) {
			Math::Vector3 dV = pos - point; // TODO: не работает с const
			float dist = std::abs(dV.length());
			double force = _constGravity * (_mass * bodyPtr->Mass()) / (dist * dist);

			point.z += force;
		}

		for (Math::Vector3& point : _line) {
			Math::Vector3 dV = pos - point; // TODO: не работает с const
			float dist = std::abs(dV.length());
			double force = _constGravity * (_mass * bodyPtr->Mass()) / (dist * dist);

			point.z += force;
		}
	}
}

void PlanePoints::Draw()
{
	ShaderGravityPoint::Instance().Use();
	Draw2::SetUniform1f(ShaderGravityPoint::u_rangeZ, 75.f);
	Draw2::SetUniform1f(ShaderGravityPoint::u_range, _spaceRange);

	static float sizePoint = 2.f;
	Draw2::SetPointSize(sizePoint);

	static float color4[] = { 0.f, 0.f, 0.0f, 0.25f };
	Draw2::SetUniform1f(ShaderGravityPoint::u_factor, 0.5f);
	Draw2::SetColorClass<ShaderGravityPoint>(color4);

	Draw2::drawPoints((float*)_points.data(), _points.size());

	Draw2::SetPointSize(1.f);
	Draw2::SetUniform1f(ShaderGravityPoint::u_factor, 0.1f);
	Draw2::drawLines((float*)_line.data(), _line.size());
}
