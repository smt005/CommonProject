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
	int _textSize = 800;
	std::string _textStr = "Wf";
	std::vector<std::string> _fonts = {"Futured.ttf", "ofont.ru_Jolly.ttf","ofont.ru_Miama Nueva.ttf","ofont.ru_Newtown.ttf","ofont.ru_Patsy Sans.ttf","ofont.ru_Radiotechnika.ttf","ofont.ru_Shablon.ttf","ofont.ru_Zekton.ttf","ofont.ru_Zing Rust.ttf","tahoma.ttf" };
	std::vector<std::string>::iterator _itFonts = _fonts.begin();
};
