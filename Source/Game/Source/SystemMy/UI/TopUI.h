#pragma once

#include "ImGuiManager/UI.h"

class SystemMy;

class TopUI final : public UI::Window {
public:
	TopUI();
	TopUI(SystemMy* systemMy);
	void OnOpen() override;
	void OnClose() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 0.f;
	float _y = 0.f;
	float _width = 0.f;
	float _height = 65.f;

	SystemMy* _systemMy = nullptr;

	int minFPS = -1;
	int FPS = 0;
	int maxFPS = -1;
};
