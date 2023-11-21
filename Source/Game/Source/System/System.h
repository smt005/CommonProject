#pragma once

#include "Game.h"
#include <map>
#include <vector>
#include <memory>
#include "glm/vec3.hpp"

class Map;
class Greed;
class Line;
class Body;

namespace Engine { class Callback; }
namespace Engine { class Text; }
class Camera;

typedef std::shared_ptr<Engine::Callback> CallbackPtr;

class System final : public Engine::Game
{
public:
	System();
	~System();
	std::filesystem::path getSourcesDir() override { return "..\\..\\Source\\Resources\\Files\\System"; }

	void init() override;
	void close() override;
	void update() override;
	void draw() override;
	void resize() override;

	void Drawline();

	void InitPhysic();
	void initCallback();
	bool load();
	void save();

private:
	Greed* _greed = nullptr;
	Greed* _greedBig = nullptr;
	Line* _interfaceLine = nullptr;
	std::vector<glm::vec3> _points;
	
	size_t _curentSunn = 0;
	std::vector<Body*> _suns;

	bool showCenter = false;
	bool showCenterMass = false;
	bool showForceVector = false;
	bool showPath = true;
	bool showRelativePath = false;

	CallbackPtr _callbackPtr;
	std::shared_ptr<Camera> _camearSide;
	std::shared_ptr<Camera> _camearTop;

public:
	static std::string _resourcesDir;
};
