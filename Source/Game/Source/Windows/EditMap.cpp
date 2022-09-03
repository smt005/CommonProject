
#include "EditMap.h"
#include "Core.h"
#include "imgui.h"

namespace Editor {

    Map::Map() {
        SetId("Edit map");
    }

    void Map::Draw() {
        _guiId = 0;

        if (ImGui::Button("Close Me")) {
            Close();
        }

    }
    void Map::OnOpen() {
        SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize);

        auto posWindow = ImGui::GetWindowPos();

        if (posWindow.x < 0.f) posWindow.x = 0.f;
        if (posWindow.y < 0.f) posWindow.y = 0.f;
        ImGui::SetWindowPos(Id().c_str(), posWindow);
        
        ImGui::SetWindowSize(Id().c_str(), { 600.f, 200.f });

        LOG("OnOpen");
    }

    void Map::OnClose() {
        LOG("OnClose");
    }
}