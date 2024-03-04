#pragma once

#include "Game.h"
#include <vector>
#include <memory>
#include "glm/vec3.hpp"
#include "UI/CommonData.h"

class Greed;
class Line;
class Camera;
class Space;

namespace Engine {
	class Callback;
}

class MySystem final : public Engine::Game {
public:
	MySystem();
	~MySystem();
	std::filesystem::path getSourcesDir() override { return "..\\..\\Source\\Resources\\Files\\System"; }

	void init() override;
	void close() override;
	void update() override;
	void draw() override;
	void resize() override;

	void Drawline();

	void initCallback();
	void Init—ameras();
	bool load();
	void save();

	void draw2(); // TEMP_

public:
	static std::shared_ptr<Space> currentSpace;

	std::shared_ptr<Engine::Callback> _callbackPtr;
	std::shared_ptr<Camera> _camearCurrent;
	std::shared_ptr<Camera> _camearSide;
	std::shared_ptr<Camera> _camearTop;
	std::shared_ptr<Camera> _camearScreen;

	Greed* _greed = nullptr;
	Greed* _greedBig = nullptr;

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
	double _time = 0;
};
