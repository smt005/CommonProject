#pragma once

#include "ImGuiManager/UI.h"
#include "Common/Help.h"
#include "ImGuiManager/Editor/Common/Common.h"
#include <vector>

class SystemMy;

class MainUI final : public UI::Window {
public:
	MainUI();
	MainUI(SystemMy* systemMy);
	void OnOpen() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 0.f;
	float _y = 0.f;
	float _width = 100.f;
	float _height = 65.f;

	int timeSpeed = 1;
	SystemMy* _systemMy = nullptr;

	int minFPS = -1;// std::numeric_limits<int>::max();
	int FPS = 0;
	int maxFPS = -1;
};
