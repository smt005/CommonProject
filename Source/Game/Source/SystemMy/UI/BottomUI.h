#pragma once

#include "ImGuiManager/UI.h"

class SystemMy;

class BottomUI final : public UI::Window {
public:
	BottomUI();
	BottomUI(SystemMy* systemMy);
	void OnOpen() override;
	void OnClose() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 0.f;
	float _y = 0.f;
	float _width = 0.f;
	float _height = 70.f;

	int timeSpeed = 1;
	SystemMy* _systemMy = nullptr;
};
