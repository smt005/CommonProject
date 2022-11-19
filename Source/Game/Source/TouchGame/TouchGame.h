#pragma once

#include "Game.h"
#include <map>
#include <vector>
#include <memory>

class Map;
class Greed;

namespace Engine { class Callback; }
typedef std::shared_ptr<Engine::Callback> CallbackPtr;

class TouchGame final : public Engine::Game
{
public:
	TouchGame();
	~TouchGame();
	std::filesystem::path getSourcesDir() override { return "..\\..\\Source\\Resources\\Files\\TouchGame"; }

	void init() override;
	void update() override;
	void draw() override;
	void resize() override;

	void CheckMouse();
	void initCallback();
	void initPhysic();
	bool load();
	void save();

	void Drawline();

public:
	static std::string _resourcesDir;

private:
	CallbackPtr _callbackPtr;
	std::vector<std::pair<std::string, std::map<std::string, bool>>> _maps;

	Greed* _greed = nullptr;

	float _force = 1.0f;

};
