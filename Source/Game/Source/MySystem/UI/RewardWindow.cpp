// ◦ Xyz ◦
#include "RewardWindow.h"
#include <imgui.h>
#include <Screen.h>
#include "../Commands/Commands.h"

RewardWindow::RewardWindow(const std::string& nextQuest, const std::string& rewardText)
    : UI::Window(this)
    , _nextQuest(nextQuest)
    , _rewardText(rewardText)
{}

void RewardWindow::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);

    float x = Engine::Screen::width() / 2.f - _width / 2.f;
    float y = Engine::Screen::height() / 2.f - _height / 2.f;

    ImGui::SetWindowPos(Id().c_str(), { x, y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void RewardWindow::OnClose() {
    CommandManager::Run(Command("SetActiveQuest", { _nextQuest, "ACTIVE" }));
}

void RewardWindow::Draw() {
    ImGui::Dummy(ImVec2(0.f, 5.f));

    ImGui::Text(_rewardText.c_str());

    ImGui::Dummy(ImVec2(0.f, 5.f));
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.f, 5.f));

    if (ImGui::Button("Next", { 180.f, 32.f})) {
        Close();
    }
}
