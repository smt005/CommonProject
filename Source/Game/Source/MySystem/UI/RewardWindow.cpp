// ◦ Xyz ◦
#include "RewardWindow.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <Screen.h>
#include "../Commands/Commands.h"

RewardWindow::RewardWindow(const std::string& currentQuest, const std::string& nextQuest, const std::string& rewardText)
	: UI::Window(this)
	, _currentQuest(currentQuest)
	, _nextQuest(nextQuest)
	, _rewardText(rewardText)
{}

void RewardWindow::OnOpen()
{
	SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);

	float x = 100.f;
	float y = 100.f;
	ImGui::SetWindowPos(Id().c_str(), { x, y });

	_screenWidth = Engine::Screen::width();
	OnResize();
}

void RewardWindow::OnClose()
{
	CommandManager::Run(Command("SetActiveQuest", { _currentQuest, "DEACTIVE" }));
	CommandManager::Run(Command("SetActiveQuest", { _nextQuest, "ACTIVE" }));
}

void RewardWindow::Update()
{
	float screenWidth = Engine::Screen::width();
	if (screenWidth != _screenWidth) {
		_screenWidth = screenWidth;
		OnResize();
	}
}

void RewardWindow::OnResize()
{
	_width = _screenWidth - 200.f;
	ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void RewardWindow::Draw()
{
	ImGui::Dummy(ImVec2(0.f, 5.f));

	ImGui::Text(_rewardText.c_str());

	ImGui::Dummy(ImVec2(0.f, 50.f));
	ImGui::Separator();

	//ImGui::BeginGroup();
	//ImGui::SameLine(_width - 200.f);
	//ImGui::Dummy(ImVec2(0.f, 4.f));
	//ImGui::SameLine(_width - 200.f);
	//ImGui::Dummy(ImVec2(0.f, 5.f));
	//ImGui::BeginChild("buttons", { _width, 40.f }, false);

	ImGui::Dummy(ImVec2(0.f, 1.f));
	//
	ImGui::Text("Complete quest.");

	ImGui::SameLine(_width - 200.f);
	if (ImGui::Button("Next", { 180.f, 32.f })) {
		Close();
	}
	//ImGui::EndChild();
	//ImGui::EndGroup();
}
