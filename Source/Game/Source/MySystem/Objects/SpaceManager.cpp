#include "SpaceManager.h"
#include "MySystem/MySystem.h"
#include "Draw/DrawLight.h"
#include "Draw/Camera/Camera.h"
#include "json/json.h"
#include "Common/Help.h"
#include "Common/Common.h"
#include "Space.h"
#include "BaseSpace.h"
#include "SpaceCpuPrototype.h"
#include "SpaceGpuPrototype.h"
#include "SpaceGpuPrototypeV3.h"
#include "SpaceV0x1.h"

void SpaceManager::AddObjectOnOrbit(Space* space, Math::Vector3d& pos, bool withAssotiation) {
	if (!space->_selectBody) {
		return;
	}

	//static std::vector<std::string> models = { "PointWhite", "PointTurquoise", "PointViolet", "PointYellow" };
	//int modelIndex = help::random_i(0, (models.size() - 1));
	//std::string model = models[modelIndex];
	
	std::string model = "BrownStone";

	float mass = help::random(10.f, 1000.f);

	Body& mainBody = *space->_selectBody;

	Math::Vector3d mainPos = mainBody.GetPos();
	Math::Vector3d gravityVector = pos - mainPos;
	Math::Vector3d normalizeGravityVector = Math::normalize(gravityVector);

	float g90 = PI / 2;
	Math::Vector3d velocity(normalizeGravityVector.x * std::cos(g90) - normalizeGravityVector.y * std::sin(g90),
		normalizeGravityVector.x * std::sin(g90) + normalizeGravityVector.y * std::cos(g90),
		0.f);

	velocity *= std::sqrtf(space->_constGravity * mainBody._mass / Math::length(gravityVector));
	velocity += mainBody._velocity;

	if (withAssotiation) {
		space->Add(model, pos, velocity, mass, "");
	} else {
		space->AddWithoutAssociation(model, pos, velocity, mass, "");
	}
}

void SpaceManager::AddObjectDirect(Space* space, Math::Vector3d& pos, Math::Vector3d& vel) {
	//...
}

void SpaceManager::AddObjects(Space* space, int count, double spaceRange, double conventionalMass) {
	Math::Vector3d gravityPos;
	double sumMass = 0;
	std::vector<Body::Ptr>& bodies = space->_bodies;

	// �������� ���
	{
		std::string model = "BrownStone";
		float mass = 50.f;

		size_t newCount = bodies.size() + count;
		bodies.reserve(newCount);

		Math::Vector3d pos;
		Math::Vector3d velocity;

		int i = 0;
		while (i < count) {
			pos.x = help::random(-spaceRange, spaceRange);
			pos.y = help::random(-spaceRange, spaceRange);
			pos.z = 0;// help::random(-spaceRange, spaceRange);

			if (pos.length() > spaceRange) {
				continue;
			}

			++i;

			//float mass = help::random(50, 150);
			space->AddWithoutAssociation(model, pos, velocity, mass, "");
		}
	}

	// ������ ������ ����
	{
		Math::Vector3d sumMassPos(0, 0, 0);

		for (Body::Ptr& bodyPtr : bodies) {
			double mass = bodyPtr->_mass;
			Math::Vector3d pos = bodyPtr->GetPos();

			sumMassPos += pos * mass;
			sumMass += mass;
		}

		gravityPos = sumMassPos / sumMass;
	}

	// ������� ���������
	{
		double mainMass = conventionalMass > 0 ? conventionalMass : sumMass / std::abs(conventionalMass);

		for (Body::Ptr& bodyPtr : bodies) {
			Math::Vector3d mainPos(0, 0, 0);
			Math::Vector3d pos = bodyPtr->GetPos();

			if (pos.x == mainPos.x && pos.y == mainPos.y && pos.z == mainPos.z) {
				continue;
			}

			Math::Vector3d gravityVector = pos - mainPos;
			Math::Vector3d normalizeGravityVector = Math::normalize(gravityVector);

			float g90 = PI / 2;
			Math::Vector3d velocity(normalizeGravityVector.x * std::cos(g90) - normalizeGravityVector.y * std::sin(g90),
				normalizeGravityVector.x * std::sin(g90) + normalizeGravityVector.y * std::cos(g90),
				0.f);

			velocity *= std::sqrtf(space->_constGravity * mainMass / Math::length(gravityVector));
			bodyPtr->_velocity = velocity;
		}
	}

	//...
	space->DataAssociation();
}

unsigned int SpaceManager::SetView(MySystem* systemMy) {
	if (systemMy->_camearCurrent == systemMy->_camearSide) {
		systemMy->_camearCurrent = systemMy->_camearTop;
		DrawLight::setClearColor(0.7f, 0.8f, 0.9f, 1.0f);
		return 1;

	}
	else if (systemMy->_camearCurrent == systemMy->_camearTop) {
		systemMy->_camearCurrent = systemMy->_camearSide;
		DrawLight::setClearColor(0.1f, 0.2f, 0.3f, 1.0f);
		return 0;
	}
}

void SpaceManager::Save(Space::Ptr space) {
	space->Save();
}

Space::Ptr SpaceManager::Load(const std::string& name) {
	std::string filePath = "Spaces/" + name + ".json";
	Json::Value valueData;

	if (!help::loadJson(filePath, valueData) || !valueData.isArray() || valueData.empty()) {
		return Space::Ptr(new Space());
	}

	std::string classStr = valueData[0]["class"].isString() ? valueData[0]["class"].asString() : std::string();

	if (classStr == Engine::GetClassName<BaseSpace>()) {
		return Space::Ptr(new BaseSpace(valueData));
	} else 
	if (classStr == Engine::GetClassName<SpaceCpuPrototype>()) {
		return Space::Ptr(new SpaceCpuPrototype(valueData));
	}
	else
	if (classStr == Engine::GetClassName<SpaceGpuPrototype>()) {
		return Space::Ptr(new SpaceGpuPrototype(valueData));
	}
	else
	if (classStr == Engine::GetClassName<SpaceGpuPrototypeV3>()) {
		return Space::Ptr(new SpaceGpuPrototypeV3(valueData));
	}
	else
	if (classStr == Engine::GetClassName<SpaceV0x1>()) {
		return Space::Ptr(new SpaceV0x1(valueData));
	}
	else {
		return Space::Ptr(new Space(valueData));
	}
	
	return Space::Ptr(new Space());
}
