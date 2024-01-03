#pragma once

#include "ImGuiManager/UI.h"
#include "Common/Help.h"
#include "ImGuiManager/Editor/Common/CommonUI.h"
#include <vector>

class MySystem;

class ListHeaviestUI final : public UI::Window {
public:
	ListHeaviestUI();
	ListHeaviestUI(MySystem* mySystem);
	void OnOpen() override;
	void OnClose() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 10.f;
	float _y = 100.f;
	float _width = 245.f;
	float _height = 500.f;

	double _time = 0;
	std::vector<std::string> _textBodies;
	MySystem* _mySystem = nullptr;
};
