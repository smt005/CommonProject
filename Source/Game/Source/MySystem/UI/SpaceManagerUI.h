#pragma once

#include "ImGuiManager/UI.h"
#include "Common/Help.h"
#include "ImGuiManager/Editor/Common/CommonUI.h"
#include <vector>

class MySystem;

class SpaceManagerUI final : public UI::Window {
public:
	SpaceManagerUI();
	SpaceManagerUI(MySystem* mySystem);
	void OnOpen() override;
	void OnClose() override;
	void Update() override;
	void Draw() override;

private:
	void CreateOrbitBody(double x, double y, double z);
	void CreateOrbitBody(double x, double y, double z, double mass, double startX, double startY, double startZ);

private:
	float _x = 10.f;
	float _y = 10.f;
	float _width = 145.f;
	float _height = 400.f;

	MySystem* _mySystem = nullptr;
};
