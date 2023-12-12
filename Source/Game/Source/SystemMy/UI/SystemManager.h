#pragma once

#include "ImGuiManager/UI.h"
#include "Common/Help.h"
#include "ImGuiManager/Editor/Common/Common.h"
#include <vector>

class SystemMy;

class SystemManager final : public UI::Window {
public:
	SystemManager();
	SystemManager(SystemMy* systemMy);
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

	SystemMy* _systemMy = nullptr;
};
