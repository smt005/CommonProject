// ◦ Xyz ◦
#pragma once

#include <ImGuiManager/UI.h>

class TopUI final : public UI::Window
{
public:
	TopUI();
	void OnOpen() override;
	void OnClose() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 0.f;
	float _y = 0.f;
	float _width = 0.f;
	float _height = 130.f; // 65.f;

	bool _showFps = false;
	float _fpsColor[4];

	int minFPS = -1;
	int FPS = 0;
	int maxFPS = -1;
};
