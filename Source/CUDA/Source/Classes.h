#pragma once
#include <vector>
#include "Wrapper.h"

namespace cuda {
	struct Vector3 {
		float x, y, z;

		Vector3() : x(0.0), y(0.0), z(0.0) {}
		Vector3(float _x, float _y, float _z) : x(_x) , y(_y) , z(_z) {}
	};

	struct Buffer {
		using uint = unsigned int;

		uint count = 0;
		std::vector<Vector3> positions;
		std::vector<float> masses;
		std::vector<Vector3> forces;
		std::vector<Vector3> velocities;

		template <typename T>
		void Load(std::vector<T>& bodies) {
			positions.clear();
			masses.clear();
			forces.clear();
			velocities.clear();

			count = (uint)bodies.size();
			if (count == 0) {
				return;
			}

			positions.reserve(count);
			masses.reserve(count);
			forces.reserve(count);

			velocities.resize(count);

			for (auto& bodyT : bodies) {
				auto pos = bodyT->GetPos();
				positions.emplace_back(Vector3(pos.x, pos.y, pos.z));

				masses.emplace_back(bodyT->_mass);
				velocities.emplace_back(Vector3(bodyT->_velocity.x, bodyT->_velocity.y, bodyT->_velocity.z));
			}
		}
	};
}
