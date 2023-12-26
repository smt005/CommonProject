#include "SpaceManager.h"
#include "MySystem/MySystem.h"
#include "Draw/DrawLight.h"
#include "Draw/Camera/Camera.h"
#include "json/json.h"
#include "Common/Help.h"
#include "Common/Common.h"
#include "Space.h"
#include "BaseSpace.h"

void SpaceManager::AddObjectOnOrbit(Space* space, Math::Vector3d& pos, bool withAssotiation) {
	std::string model = "BrownStone";
	float mass = 10.f;

	if (!space->_selectBody) {
		return;
	}

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
		space->Add("BrownStone", pos, velocity, mass, "");
	} else {
		space->AddWithoutAssociation("BrownStone", pos, velocity, mass, "");
	}
}

void SpaceManager::AddObjectDirect(Space* space, Math::Vector3d& pos, Math::Vector3d& vel) {
	//...
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
	} else {
		return Space::Ptr(new Space(valueData));
	}

	return Space::Ptr(new Space());
}
