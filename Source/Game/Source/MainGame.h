#pragma once

#include "Game.h"
#include <map>
#include <vector>
#include <memory>

class Map;

namespace Engine { class Callback; }
typedef std::shared_ptr<Engine::Callback> CallbackPtr;

class MainGame final : public Engine::Game
{
	enum class State {
		MENU,
		GAME,
		EXIT
	};

public:
	MainGame();
	~MainGame();
	std::filesystem::path getSourcesDir() override { return "..\\..\\Source\\Resources\\Files"; }

	void init() override;
	void update() override;
	void draw() override;
	void resize() override;

	void initCallback();
	void initPhysic();
	bool load();
	void save();

	Map& currentMap();
	void changeMap(const bool right);
	void hit(const int x, const int y, const bool action = false);

public:
	static std::string _resourcesDir;

private:
	CallbackPtr _callbackPtr;
	std::vector<std::pair<std::string, std::map<std::string, bool>>> _maps;
	int _indexCurrentMap;
	double _updateTime;
	State _state;
};
