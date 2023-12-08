
#include "SpatialGrid.h"
#include <memory>
#include <cmath>
#include <glm/ext/scalar_constants.hpp>


SpatialGrid::SpatialGrid() {
	if (false) {
		_points.reserve(2);
		_points.emplace_back(1000, 1000, 0);
		_points.emplace_back(10000, 10000, 0);
	}
	else
	{
		// 1000.f, 1000.f, 10, { 0.9125f, 0.125f, 0.5f, 0.9125f }, 2.f
		//float minRadius, float distans, float count, const float(&color)[4], float widthLine

		{
			float distans = 1000.f;
			float count = 11;

			for (float index = 1; index < count; index += 1.f) {
				float devideCount = 31.f;
				float dAngle = glm::pi<float>() / devideCount;
				float sumAngle = glm::pi<float>() * 2.f;
				float radius = distans * index;

				for (float angle = 0.f; angle < (sumAngle + dAngle); angle += dAngle) {
					float x = radius * std::cos(angle) - radius * std::sin(angle);
					float y = radius * std::sin(angle) + radius * std::cos(angle);

					_points.emplace_back(x, y, 0.f);
				}
			}
		}

		//...
		{
			float distans = 10000.f;
			float count = 11;

			for (float index = 1; index < count; index += 1.f) {
				float devideCount = 31.f;
				float dAngle = glm::pi<float>() / devideCount;
				float sumAngle = glm::pi<float>() * 2.f;
				float radius = distans * index;

				for (float angle = 0.f; angle < (sumAngle + dAngle); angle += dAngle) {
					float x = radius * std::cos(angle) - radius * std::sin(angle);
					float y = radius * std::sin(angle) + radius * std::cos(angle);

					_points.emplace_back(x, y, 0.f);
				}
			}
		}

	}
}
