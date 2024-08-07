// ◦ Xyz ◦

#include "SpaceManager.h"
#include <MySystem/MySystem.h>
#include <Draw/DrawLight.h>
#include <Draw/Camera/Camera.h>
#include <json/json.h>
#include <Common/Help.h>
#include <Common/Common.h>
#include "Space.h"
#include "BaseSpace.h"
#include "SpaceGpuX0.h"
#include "SpaceGpuX1.h"
#include "SpaceTree00.h"
#include "SpaceTree01.h"
#include "SpaceTree02.h"

void SpaceManager::AddObjectOnOrbit(Space* space, Math::Vector3& pos, bool withAssotiation)
{
	if (space->_bodies.empty()) {
		return;
	}

	if (!space->_selectBody) {
		if (space->_bodies.empty() && space->_bodies.front()) {
			return;
		}

		space->_selectBody = space->_bodies.front();
	}

	//static std::vector<std::string> models = { "PointWhite", "PointTurquoise", "PointViolet", "PointYellow" };
	//int modelIndex = help::random_i(0, (models.size() - 1));
	//std::string model = models[modelIndex];

	std::string model = "BrownStone";

	float minMass = 100.f;
	std::string minMassStr = space->_params["MIN_MASS"];
	if (!minMassStr.empty()) {
		minMassStr = std::stoi(minMassStr);
	}

	float maxMass = 1000.f;
	std::string maxMassStr = space->_params["MAX_MASS"];
	if (!maxMassStr.empty()) {
		maxMass = std::stoi(maxMassStr);
	}

	float mass = help::random(minMass, maxMass);

	Body& mainBody = *space->_selectBody;

	Math::Vector3 mainPos = mainBody.GetPos();
	Math::Vector3 gravityVector = pos - mainPos;
	Math::Vector3 normalizeGravityVector = Math::normalize(gravityVector);

	float g90 = PI / 2;
	Math::Vector3 velocity(normalizeGravityVector.x * std::cos(g90) - normalizeGravityVector.y * std::sin(g90),
		normalizeGravityVector.x * std::sin(g90) + normalizeGravityVector.y * std::cos(g90),
		0.f);

	velocity *= std::sqrtf(space->_constGravity * mainBody.Mass() / Math::length(gravityVector));
	velocity += mainBody.Velocity();

	float speedFactor = 1.f;
	std::string speedFactorStr = space->_params["SPEED_FACTOR"];
	if (!speedFactorStr.empty()) {
		speedFactor = std::stoi(speedFactorStr);
		velocity *= speedFactor;
	}

	space->Add<BodyData>(model, pos, velocity, mass, "");

	if (withAssotiation) {
		space->Preparation();
	}
}

void SpaceManager::AddObjectDirect(Space* space, Math::Vector3& pos, Math::Vector3& vel)
{
	//...
}

void SpaceManager::AddObjects(Space* space, int count, double spaceRange, double conventionalMass)
{
	Math::Vector3 gravityPos;
	double sumMass = 0;
	std::vector<Body::Ptr>& bodies = space->_bodies;

	// Создание тел
	{
		std::string model = "BrownStone";
		float mass = 50.f;

		size_t newCount = bodies.size() + count;
		bodies.reserve(newCount);

		Math::Vector3 pos;
		Math::Vector3 velocity;

		int i = 0;
		while (i < count) {
			pos.x = help::random(-spaceRange, spaceRange);
			pos.y = help::random(-spaceRange, spaceRange);
			pos.z = 0.f; // help::random(-spaceRange, spaceRange);

			if (pos.length() > spaceRange) {
				continue;
			}

			++i;

			//float mass = help::random(50, 150);
			space->Add<BodyData>(model, pos, velocity, mass, "");
		}

		space->Preparation();
	}

	// Расчёт центра масс
	{
		Math::Vector3 sumMassPos(0, 0, 0);

		for (Body::Ptr& bodyPtr : bodies) {
			double mass = bodyPtr->Mass();
			Math::Vector3 pos = bodyPtr->GetPos();

			sumMassPos += pos * mass;
			sumMass += mass;
		}

		gravityPos = sumMassPos / sumMass;
	}

	// Рассчёт скоростей
	{
		double mainMass = conventionalMass > 0 ? conventionalMass : sumMass / std::abs(conventionalMass);

		for (Body::Ptr& bodyPtr : bodies) {
			Math::Vector3 mainPos(0, 0, 0);
			Math::Vector3 pos = bodyPtr->GetPos();

			if (pos.x == mainPos.x && pos.y == mainPos.y && pos.z == mainPos.z) {
				continue;
			}

			Math::Vector3 gravityVector = pos - mainPos;
			Math::Vector3 normalizeGravityVector = Math::normalize(gravityVector);

			float g90 = PI / 2;
			Math::Vector3 velocity(normalizeGravityVector.x * std::cos(g90) - normalizeGravityVector.y * std::sin(g90),
				normalizeGravityVector.x * std::sin(g90) + normalizeGravityVector.y * std::cos(g90),
				0.f);

			velocity *= std::sqrtf(space->_constGravity * mainMass / Math::length(gravityVector));
			bodyPtr->Velocity() = velocity;
		}
	}

	//...
	space->Preparation();
}

void SpaceManager::AddObject(const std::string& nameModel, const Math::Vector3& pos, const Math::Vector3& vel, float mass, const Color& color)
{
	if (!MySystem::currentSpace) {
		return;
	}

	const std::string& model = !nameModel.empty() ? nameModel : "BrownStone";

	Body& body = MySystem::currentSpace->Add<BodyData>(nameModel, pos, vel, mass, "");
	body.color = color;

	MySystem::currentSpace->Preparation();
}

unsigned int SpaceManager::SetView(MySystem* systemMy)
{
	if (!systemMy) {
		return 0;
	}

	if (systemMy->_camearCurrent && systemMy->_camearCurrent == systemMy->_camearSide) {
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

void SpaceManager::Save(Space::Ptr space)
{
	space->Save();
}

Space::Ptr SpaceManager::Load(const std::string& name)
{
	std::string filePath = "Spaces/" + name + ".json";
	Json::Value valueData;

	if (!help::loadJson(filePath, valueData) || !valueData.isArray() || valueData.empty()) {
		return Space::Ptr(new Space());
	}

	std::string classStr = valueData[0]["class"].isString() ? valueData[0]["class"].asString() : std::string();

	if (classStr == Engine::GetClassName<BaseSpace>()) {
		return Space::Ptr(new BaseSpace(valueData));
	}
	else if (classStr == Engine::GetClassName<SpaceGpuX0>()) {
		return Space::Ptr(new SpaceGpuX0(valueData));
	}
	else if (classStr == Engine::GetClassName<SpaceGpuX1>()) {
		return Space::Ptr(new SpaceGpuX1(valueData));
	}
	else if (classStr == Engine::GetClassName<SpaceTree00>()) {
		return Space::Ptr(new SpaceTree00(valueData));
	}
	else if (classStr == Engine::GetClassName<SpaceTree01>()) {
		return Space::Ptr(new SpaceTree01(valueData));
	}
	else if (classStr == Engine::GetClassName<SpaceTree02>()) {
		return Space::Ptr(new SpaceTree02(valueData));
	}
	else {
		return Space::Ptr(new Space(valueData));
	}

	return Space::Ptr(new Space());
}

const std::vector<std::string>& SpaceManager::GetListClasses()
{
	static const std::vector<std::string> listClasses = { "SpaceTree02", "SpaceGpuX0", "SpaceTree01", "SpaceGpuX1", "BaseSpace", "SpaceTree00" };
	return listClasses;
}

std::shared_ptr<Space> SpaceManager::CopySpace(const std::string& className, Space* space)
{
	if (!space) {
		return Space::Ptr(new Space("DEFAULT"));
	}

	std::shared_ptr<Space> copySpacePtr;

	if (className == Engine::GetClassName<BaseSpace>()) {
		copySpacePtr = std::make_shared<BaseSpace>();
	}
	else if (className == Engine::GetClassName<SpaceGpuX0>()) {
		copySpacePtr = std::make_shared<SpaceGpuX0>();
	}
	else if (className == Engine::GetClassName<SpaceGpuX1>()) {
		copySpacePtr = std::make_shared<SpaceGpuX1>();
	}
	else if (className == Engine::GetClassName<SpaceTree00>()) {
		copySpacePtr = std::make_shared<SpaceTree00>();
	}
	else if (className == Engine::GetClassName<SpaceTree01>()) {
		copySpacePtr = std::make_shared<SpaceTree01>();
	}
	else if (className == Engine::GetClassName<SpaceTree02>()) {
		copySpacePtr = std::make_shared<SpaceTree02>();
	}

	auto* copySpace = copySpacePtr.get();

	copySpace->_name = space->_name;
	copySpace->_constGravity = space->_constGravity;

	copySpace->deltaTime = space->deltaTime;
	copySpace->countOfIteration = space->countOfIteration;
	copySpace->timePassed = space->timePassed;
	copySpace->processGPU = space->processGPU;
	copySpace->multithread = space->multithread;
	copySpace->tag = space->tag;

	copySpace->_params = space->_params;
	copySpace->_bodies = space->_bodies;

	//SpatialGrid spatialGrid;
	copySpace->_skyboxModel = space->_skyboxModel;

	return copySpacePtr;
}
