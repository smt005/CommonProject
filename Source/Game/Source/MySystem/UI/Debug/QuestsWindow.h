#pragma once

#include "ImGuiManager/UI.h"

class QuestsWindow final : public UI::Window {
public:
	QuestsWindow();
	void OnOpen() override;
	void Draw() override;

private:
	float _width = 200.f;
	float _height = 500.f;

	std::vector<std::pair<std::string, std::string>> _commands;
};
