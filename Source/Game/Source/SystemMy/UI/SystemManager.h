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
	void Update() override;
	void Draw() override;

private:
	float _x = 10.f;
	float _y = 10.f;
	float _width = 145.f;
	float _height = 160.f;

	SystemMy* _systemMy = nullptr;
};
