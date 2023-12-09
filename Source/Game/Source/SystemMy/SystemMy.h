#pragma once

#include "Game.h"
#include <map>
#include <vector>
#include <memory>
#include "glm/vec3.hpp"
#include "Objects/SystemClass.h"

class Map;

class Greed;
class Line;
class BodyMy;
class TopUI;
class BottomUI;
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

	bool PerspectiveView() {
		return _perspectiveView;
	}

	void SetPerspectiveView(bool perspectiveView);

	bool ViewByObject() {
		return _viewByObject;
	}

	void SetViewByObject(bool viewByObject);
	void NormalizeSystem();

private:
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

	CallbackPtr _callbackPtr;
	std::shared_ptr<Camera> _camearSide;
	std::shared_ptr<Camera> _camearTop;

	struct LockMouse {
		bool lockPinch = false;
		bool lockAllPinch = false;
		float bottomHeight = 60.f;

		bool IsLock() {
			return lockPinch || lockAllPinch;
		}

	} _lockMouse;

	int _orbite = 0;
	int _timeSpeed = 1;
	bool _viewByObject = false;
	bool _perspectiveView = true;

public:
	static std::string _resourcesDir;
	static double time;
};
