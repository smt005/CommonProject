#pragma once

#include "ImGuiManager/UI.h"

class MySystem;

class TopUI final : public UI::Window {
public:
	TopUI();
	TopUI(MySystem* mySystem);
	void OnOpen() override;
	void OnClose() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 0.f;
	float _y = 0.f;
	float _width = 0.f;
	float _height = 65.f;

	MySystem* _mySystem = nullptr;

	int minFPS = -1;
	int FPS = 0;
	int maxFPS = -1;
};
