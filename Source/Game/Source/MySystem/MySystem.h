// ◦ Xyz ◦
#pragma once

#include <Game.h>
#include <memory>

class Camera;
class Space;

namespace Engine
{
	class Callback;
}

class MySystem final : public Engine::Game
{
public:
	MySystem() = default;
	~MySystem() = default;
	std::filesystem::path getSourcesDir() override { return "..\\..\\Source\\Resources\\Files\\System"; }

	void init() override;
	void close() override;
	void update() override;
	void draw() override;
	void resize() override;

	void initCallback();
	void InitСameras();
	bool load();
	void save();

public:
	std::shared_ptr<Engine::Callback> _callbackPtr;
	std::shared_ptr<Camera> _camearCurrent;
	static std::shared_ptr<Camera> _camearSide;
	std::shared_ptr<Camera> _camearTop;
	std::shared_ptr<Camera> _camearScreen;

private:
	double _time = 0;

public:
	static std::shared_ptr<Space> currentSpace;
	static bool gravitypoints;
};
