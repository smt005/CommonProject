#pragma once

#include "Game.h"
#include <map>
#include <vector>
#include <memory>

class Map;
class Greed;

namespace Engine { class Callback; }
namespace Engine { class Text; }

typedef std::shared_ptr<Engine::Callback> CallbackPtr;

class TouchGame final : public Engine::Game
{
public:
	TouchGame();
	~TouchGame();
	std::filesystem::path getSourcesDir() override { return "..\\..\\Source\\Resources\\Files\\TouchGame"; }

	void init() override;
	void close() override;
	void update() override;
	void draw() override;
	void resize() override;

	void CheckMouse();
	void initCallback();
	void initPhysic();
	bool load();
	bool loadCamera();
	void save();

	void Drawline();
	void DrawText();

	void MakeGreed();
	void RemoveGreed();

public:
	static std::string _resourcesDir;

private:
	CallbackPtr _callbackPtr;
	std::string _currentGamMap;
	Greed* _greed = nullptr;
	Engine::Text* _text = nullptr;
};
