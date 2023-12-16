#pragma once

#include "Game.h"
#include <map>
#include <vector>
#include <memory>
#include "glm/vec3.hpp"
#include "UI/CommonData.h"
#include "Objects/SystemClass.h"
#include "Math/Vector.h"

class Map;

class Greed;
class Line;
class BodyMy;
class TopUI;
class BottomUI;
class ListHeaviestUI;
class SystemManager;

namespace Engine { class Callback; }
namespace Engine { class Text; }
class Camera;

typedef std::shared_ptr<Engine::Callback> CallbackPtr;

class SystemMy final : public Engine::Game
{
public:
	friend TopUI;
	friend BottomUI;
	friend ListHeaviestUI;
	friend SystemManager;

	SystemMy();
	~SystemMy();
	std::filesystem::path getSourcesDir() override { return "..\\..\\Source\\Resources\\Files\\System"; }

	void init() override;
	void close() override;
	void update() override;
	void draw() override;
	void resize() override;

	void Drawline();

	void initCallback();
	bool load();
	void save();

	int GetOrbite() const {
		return _orbite;
	}
	void SetOrbite(int orbite) {
		_orbite = orbite;
	}

	void SetTimeSpeed(int timeSpeed) {
		_timeSpeed = timeSpeed;
	}

private:
public:
	std::shared_ptr<SystemMap> _systemMap;

	Greed* _greed = nullptr;
	Greed* _greedBig = nullptr;
	Line* _interfaceLine = nullptr;
	std::vector<glm::vec3> _points;

	bool showCenter = false;
	bool showCenterMass = false;
	bool showForceVector = false;
	bool showPath = true;
	bool showRelativePath = false;

	bool _orbite = true;
	int _timeSpeed = 1;

	CallbackPtr _callbackPtr;
	std::shared_ptr<Camera> _camearSide;
	std::shared_ptr<Camera> _camearScreen;
	Math::Vector3d focusToo;

	struct LockMouse {
		bool lockPinch = false;
		bool lockAllPinch = false;
		float bottomHeight = 60.f;

		bool IsLock() {
			return lockPinch || (CommonData::lockAction > 0);
		}

	} _lockMouse;

public:
	static std::string _resourcesDir;
	static double time;
};
