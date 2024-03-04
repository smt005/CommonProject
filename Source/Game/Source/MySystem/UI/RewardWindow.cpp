#include "RewardWindow.h"
#include <imgui.h>
#include <Screen.h>

RewardWindow::RewardWindow() : UI::Window(this) { }

void RewardWindow::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    float x = Engine::Screen::width() / 2.f - _width / 2.f;
    float y = Engine::Screen::height() / 2.f - _height / 2.f;

    ImGui::SetWindowPos(Id().c_str(), { x, y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void RewardWindow::Draw() {
    ImGui::Text("Test text.");

    ImGui::Separator();

    if (ImGui::Button("Close", { 128.f, 32.f})) {
        Close();
    }
}
