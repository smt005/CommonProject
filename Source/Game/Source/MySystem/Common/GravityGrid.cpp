// ◦ Xyz ◦

#include "GravityGrid.h"
#include <cmath>
#include <glm/mat4x4.hpp>
#include <Draw2/Draw2.h>
#include <Draw/Camera/Camera.h>
#include "ShaderGravityGrid.h"
#include <Common/Help.h>

#include <MySystem/MySystem.h>
#include <MySystem/Objects/Body.h>
#include <MySystem/Objects/Space.h>

GravityGrid* GravityGrid::gravityGridPtr = nullptr;

void GravityGrid::Init(float spaceRange, float offset)
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

	if (!ShaderGravityGrid::Instance().Inited()) {
		ShaderGravityGrid::Instance().Init("GravityGrid.vert", "GravityGrid.frag");
	}
}

void GravityGrid::Draw()
{
	ShaderGravityGrid::Instance().Use();

	Draw2::SetUniform1f(ShaderGravityGrid::u_rangeZ, 75.f);
	Draw2::SetUniform1f(ShaderGravityGrid::u_range, _spaceRange);

	float color4[] = { 1.f, 1.f, 1.0f, 1.f };
	Draw2::SetColorClass<ShaderGravityGrid>(color4);

	const int countBodies = MySystem::currentSpace ? MySystem::currentSpace->_bodies.size() : 0;
	Draw2::SetUniform1i(ShaderGravityGrid::u_body_count, countBodies);

	if (countBodies) {
		float* bodiesPos = new float[countBodies * 3];
		float* bodiesColor = new float[countBodies * 3];
		float* bodiesMass = new float[countBodies];

		int indexBodies = 0;
		int indexColors = 0;
		int indexMasses = 0;

		for (auto& bodyPtr : MySystem::currentSpace->_bodies) {
			auto pos = bodyPtr->GetPos();
			bodiesPos[indexBodies++] = pos.x;
			bodiesPos[indexBodies++] = pos.y;
			bodiesPos[indexBodies++] = pos.z;

			bodiesColor[indexColors++] = bodyPtr->color.getRed();
			bodiesColor[indexColors++] = bodyPtr->color.getGreen();
			bodiesColor[indexColors++] = bodyPtr->color.getBlue();

			bodiesMass[indexMasses++] = bodyPtr->Mass();
		}

		Draw2::SetUniform3fv(ShaderGravityGrid::u_body_positions, bodiesPos, countBodies);
		Draw2::SetUniform1fv(ShaderGravityGrid::u_body_massess, bodiesMass, countBodies);
		Draw2::SetUniform3fv(ShaderGravityGrid::u_body_colors, bodiesColor, countBodies);

		delete[] bodiesPos;
		delete[] bodiesColor;
		delete[] bodiesMass;

		if (_mass > 1.f) {
			float k = _mass / 10.f;
			_mass -= k;
		}

		Draw2::SetUniform1f(ShaderGravityGrid::u_mass_factor, _mass);
	}

	if (_splashCount > 0) {
		float* splashPosition = reinterpret_cast<float*>(_splashPosition.data());
		float* distances = reinterpret_cast<float*>(_distances.data());

		Draw2::SetUniform1fv(ShaderGravityGrid::u_distances, distances, _splashCount);
		Draw2::SetUniform3fv(ShaderGravityGrid::u_splashPosition, splashPosition, _splashCount);
	}
	Draw2::SetUniform1i(ShaderGravityGrid::u_splashCount, _splashCount);

	Draw2::SetPointSize(-1.f);
	Draw2::SetUniform1f(ShaderGravityGrid::u_factor, 0.75f);
	Draw2::drawPoints((float*)_points.data(), _points.size());

	Draw2::SetPointSize(1.f);
	Draw2::SetUniform1f(ShaderGravityGrid::u_factor, 0.125f);
	Draw2::drawLines((float*)_line.data(), _line.size());
}

void GravityGrid::Update(double dt)
{
	if (!_splashPosition.empty() && !_distances.empty()) {
		int index = 0;
		while (_distances[index] > 500.f) {
			auto itPos = _splashPosition.begin() + index;
			if (itPos != _splashPosition.end()) {
				_splashPosition.erase(itPos);
			}

			auto itTime = _distances.begin() + index;
			if (itTime != _distances.end()) {
				_distances.erase(itTime);
			}

			_splashCount = _splashPosition.size();
			break;
		}

		for (float& dist : _distances) {
			dist += static_cast<float>(dt) * 400.f;
		}
	}
}

void GravityGrid::AddTime(const Math::Vector3& pos, float dist)
{
	_splashPosition.emplace_back(pos);
	_distances.emplace_back(dist);

	_splashCount = _splashPosition.size();
}
