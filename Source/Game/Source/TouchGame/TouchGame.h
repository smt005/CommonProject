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
	enum class State {
		MENU,
		GAME,
		EXIT
	};

public:
	TouchGame();
	~TouchGame();
	std::filesystem::path getSourcesDir() override { return "..\\..\\Source\\Resources\\Files\\TouchGame"; }

	void init() override;
	void update() override;
	void draw() override;
	void resize() override;

	void initCallback();
	void CheckMouse();
	void initPhysic();
	bool load();
	void save();

	Map& currentMap();
	void changeMap(const bool right);
	void hit(const int x, const int y, const bool action = false);
	void Drawline();

public:
	static std::string _resourcesDir;

private:
	CallbackPtr _callbackPtr;
	std::vector<std::pair<std::string, std::map<std::string, bool>>> _maps;
	int _indexCurrentMap;
	double _updateTime;
	State _state;
	float _mousePos[2];
	float _cameraSpeed = 1.f;

	Greed* _greed = nullptr;
	float _lenghtNormal = 10.f;
	float _widthNormal = 1.f;
	bool _qwe0_ = true;
	bool _qwe_ = true;
	bool _type_ = false;

	std::string _editMapWindow;
};
