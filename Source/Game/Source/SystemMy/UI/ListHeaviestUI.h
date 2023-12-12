#pragma once

#include "ImGuiManager/UI.h"
#include "Common/Help.h"
#include "ImGuiManager/Editor/Common/Common.h"
#include <vector>

class SystemMy;

class ListHeaviestUI final : public UI::Window {
public:
	ListHeaviestUI();
	ListHeaviestUI(SystemMy* systemMy);
	void OnOpen() override;
	void OnClose() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 10.f;
	float _y = 100.f;
	float _width = 145.f;
	float _height = 390.f;

	double _time = 0;
	std::vector<std::string> _textBodies;
	SystemMy* _systemMy = nullptr;
};
