// ◦ Xyz ◦
#pragma once

#include "ImGuiManager/UI.h"
#include "Common/Help.h"
#include "ImGuiManager/Editor/Common/CommonUI.h"
#include <vector>

class MySystem;

class ComputationsUI final : public UI::Window {
public:
	ComputationsUI();
	void OnOpen() override;
	void OnClose() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 160.f;
	float _y = 30.f;
	float _width = 145.f;
	float _height = 600.f;

	MySystem* _mySystem = nullptr;
};
