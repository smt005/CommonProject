#include "SpaceManager.h"
#include "SystemMy/SystemMy.h"
#include "Draw/DrawLight.h"
#include "Draw/Camera/Camera.h"

void SpaceManager::AddObjectOnOrbit(SystemMap* space, Math::Vector3d& pos) {
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
	space->Add("BrownStone", pos, velocity, mass, "");
}

void SpaceManager::AddObjectDirect(Space* space, Math::Vector3d& pos, Math::Vector3d& vel) {
	//...
}

unsigned int SpaceManager::SetView(SystemMy* systemMy) {
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