// ◦ Xyz ◦
#pragma once

#include "ImGuiManager/UI.h"

class RewardWindow final : public UI::Window {
public:
	RewardWindow(const std::string& nextQuest, const std::string& rewardText);
	void OnOpen() override;
	void OnClose() override;
	void Draw() override;

private:
	float _width = 200.f;
	float _height = 150.f;
	std::string _nextQuest;
	std::string _rewardText;
};
