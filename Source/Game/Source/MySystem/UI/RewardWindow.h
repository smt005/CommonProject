#pragma once

#include "ImGuiManager/UI.h"

class RewardWindow final : public UI::Window {
public:
	RewardWindow();
	void OnOpen() override;
	void Draw() override;

private:
	float _width = 200.f;
	float _height = 100.f;
};
