#include "SpaceCpuPrototype.h"
#include <thread>

#define NEW_FUN_TAG 0;

namespace  {
	void GetForce(int count, float* masses, float* positionsX, float* positionsY, float* forcesX, float* forcesY) {
		double _constGravity = 0.01f;
		int statIndex = 0;
		int endIndex = count;
		int sizeData = count;

		for (int index = statIndex; index < endIndex; ++index) {
			for (size_t otherIndex = 0; otherIndex < sizeData; ++otherIndex) {
				if (index == otherIndex) {
					continue;
				}

				float gravityVecX = positionsX[otherIndex] - positionsX[index];
				float gravityVecY = positionsY[otherIndex] - positionsY[index];

				double dist = sqrt(gravityVecX * gravityVecX + gravityVecY * gravityVecY);
				gravityVecX /= dist;
				gravityVecY /= dist;

				double force = _constGravity * (masses[index] * masses[otherIndex]) / (dist * dist);
				gravityVecX *= force;
				gravityVecY *= force;

				forcesX[index] += gravityVecX;
				forcesY[index] += gravityVecY;
			}
		}
	}
}

void SpaceCpuPrototype::Update(double dt) {
	size_t sizeData = _datas.size();
	if (sizeData <= 1) {
		return;
	}

#if NEW_FUN_TAG

	int count = _datas.size();
	float* masses = new float[count];
	float* positionsX = new float[count];
	float* positionsY = new float[count];
	float* forcesX = new float[count];
	float* forcesY = new float[count];

	for (size_t index = 0; index < count; ++index) {
		Body::Data& data = _datas[index];

		masses[index] = data.mass;

		positionsX[index] = data.pos.x;
		positionsY[index] = data.pos.y;

		forcesX[index] = 0.f;
		forcesY[index] = 0.f;
	}

	GetForce(count, masses, positionsX, positionsY, forcesX, forcesY);

#else

	auto getForce = [&](size_t statIndex, size_t endIndex) {
		for (size_t index = statIndex; index <= endIndex; ++index) {
			Body::Data& data = _datas[index];

			float radius = _bodies[index]->_scale;

			double mass = data.mass;
			Math::Vector3d& pos = data.pos;
			Math::Vector3d& forceVec = data.force;
			forceVec.x = 0;
			forceVec.y = 0;
			forceVec.z = 0;

			for (size_t otherIndex = 0; otherIndex < sizeData; ++otherIndex) {
				Body::Data& otherBody = _datas[otherIndex];

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
			}
		}
	};

	/*unsigned int counThread = static_cast<double>(thread::hardware_concurrency());
	int lastIndex = _bodies.size() - 1;

	if (threadEnable && ((lastIndex * 2) > counThread)) {
		double counThreadD = static_cast<double>(counThread);

		double lastIndexD = static_cast<double>(lastIndex);
		double dSizeD = lastIndexD / counThreadD;
		int dSize = static_cast<int>(round(dSizeD));
		dSize = dSize == 0 ? 1 : dSize;

		vector<std::pair<size_t, size_t>> ranges;
		vector<std::thread> threads;
		threads.reserve(counThread);

		int statIndex = 0; size_t endIndex = statIndex + dSize;
		while(statIndex < lastIndex) {
			ranges.emplace_back(statIndex, endIndex);
			statIndex = ++endIndex; endIndex = statIndex + dSize;
		}

		ranges.back().second = lastIndex;
		for (auto& pair : ranges) {
			threads.emplace_back([&]() {
				getForce(pair.first, pair.second);
			});
		}

		for (thread& th : threads) {
			th.join();
		}
	} else*/
	{
		getForce(0, _bodies.size() - 1);
	}

#endif

	// ...
	float longÂistanceFromStar = 150000.f;
	size_t needDataAssociation = std::numeric_limits<double>::min();
	std::vector<size_t> indRem;

	size_t size = _bodies.size();

	Body::Ptr star = GetHeaviestBody();
	Math::Vector3d posStar = star ? star->GetPos() : Math::Vector3d();

	for (size_t index = 0; index < size; ++index) {
		Body::Ptr& body = _bodies[index];
		if (!body) {
			continue;
		}

		static double minForce = std::numeric_limits<double>::min();
		if ((body->_dataPtr->force.length() < minForce) && (star && (posStar - body->GetPos()).length() > longÂistanceFromStar)) {
			indRem.emplace_back(index);
			++needDataAssociation;
			continue;
		}

#if NEW_FUN_TAG
		Math::Vector3d newForce(forcesX[index], forcesY[index], 0.0);
		Math::Vector3d acceleration = newForce / body->_mass;
#else
		Math::Vector3d acceleration = body->_dataPtr->force / body->_mass;
#endif

		Math::Vector3d newVelocity = acceleration * static_cast<double>(dt);

		body->_velocity += newVelocity;

		body->_dataPtr->pos += body->_velocity * static_cast<double>(dt);
		body->SetPos(body->_dataPtr->pos);

		body->force = body->_dataPtr->force.length();
	}

#if NEW_FUN_TAG
	delete[] masses;
	delete[] positionsX;
	delete[] positionsY;
	delete[] forcesX;
	delete[] forcesY;
#endif
	//...
	//DataAssociation();

	//...
	if (dt > 0) {
		++time;
	}
	else {
		--time;
	}
}