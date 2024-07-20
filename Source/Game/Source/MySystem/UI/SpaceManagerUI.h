// ◦ Xyz ◦
#pragma once

#include <ImGuiManager/UI.h>
#include <Common/Help.h>
#include <ImGuiManager/Editor/Common/CommonUI.h>
#include <vector>

class SpaceManagerUI final : public UI::Window
{
public:
	SpaceManagerUI()
		: UI::Window(this)
	{}
	void OnOpen() override;
	void OnClose() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 10.f;
	float _y = 30.f;
	float _width = 145.f;
	float _height = 600.f;
};
