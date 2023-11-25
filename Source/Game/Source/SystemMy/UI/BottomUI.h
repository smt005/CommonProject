#pragma once

#include "ImGuiManager/UI.h"
#include "Common/Help.h"
#include "ImGuiManager/Editor/Common/Common.h"
#include <vector>

class SystemMy;

class BottomUI final : public UI::Window {
public:
	BottomUI(/*SystemMy* mystemMy*/);
	void OnOpen() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 0.f;
	float _y = 0.f;
	float _width = 100.f;
	float _height = 30.f;

	//SystemMy* _mystemMy = nullptr;
};
