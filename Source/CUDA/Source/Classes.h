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

		struct Pair {
			unsigned int first = 0;
			unsigned int second = 0;
		};

		uint count = 0;
		std::vector<Vector3> positions;
		std::vector<float> radiuses;
		std::vector<float> masses;
		std::vector<Vector3> forces;
		std::vector<Vector3> velocities;

		unsigned int countCollisions = 0;
		std::vector<Pair> collisions;

		template <typename T>
		void Load(std::vector<T>& bodies) {
			positions.clear();
			radiuses.clear();
			masses.clear();
			forces.clear();
			velocities.clear();
			collisions.clear();

			count = (uint)bodies.size();
			if (count == 0) {
				return;
			}

			positions.reserve(count);
			masses.reserve(count);
			velocities.reserve(count);

			forces.resize(count);
			
			countCollisions = 0;
			collisions.resize(count);

			for (auto& bodyT : bodies) {
				auto pos = bodyT->GetPos();
				positions.emplace_back(Vector3(pos.x, pos.y, pos.z));

				radiuses.emplace_back(bodyT->_scale);
				masses.emplace_back(bodyT->_mass);

				float vx = bodyT->_velocity.x;
				float vy = bodyT->_velocity.y;
				float vz = bodyT->_velocity.z;
				velocities.emplace_back(vx, vy, vz);
			}
		}

		void Reset() {
			forces.clear();
			forces.resize(count);

			countCollisions = 0;
		}
	};
}
