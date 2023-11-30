
#include "SystemManager.h"
#include "imgui.h"

#include "../SaveManager.h"

SystemManager::SystemManager() {
    SetId("SystemManager");
    Close();
}

SystemManager::SystemManager(SystemMy* systemMy)
    : UI::Window()
    , _systemMy(systemMy)
{
    SetId("SystemManager");
}

void SystemManager::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void SystemManager::Update() {

}

void SystemManager::Draw() {
    //ImGuiStyle& style = ImGui::GetStyle();
    //style.FramePadding.y = 3.f;

    if (ImGui::Button("Save", { 128.f, 32.f })) {
        SaveManager::Save();
    }
    
    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Reload", { 128.f, 32.f })) {
        SaveManager::Reload();
    }
}
