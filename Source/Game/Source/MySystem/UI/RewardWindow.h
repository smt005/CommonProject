// ◦ Xyz ◦
#pragma once

#include <ImGuiManager/UI.h>

class RewardWindow final : public UI::Window
{
public:
	RewardWindow(const std::string& currentQuest, const std::string& nextQuest, const std::string& rewardText);
	void OnOpen() override;
	void OnClose() override;
	void Update() override;
	void Draw() override;

	void OnResize();

private:
	float _width = 200.f;
	float _height = 150.f;
	float _screenWidth = 0.f;

	std::string _currentQuest;
	std::string _nextQuest;
	std::string _rewardText;
};
