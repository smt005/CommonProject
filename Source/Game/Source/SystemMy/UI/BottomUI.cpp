
#include "BottomUI.h"
#include "imgui.h"

#include "Screen.h"
#include "../SystemMy.h"
#include "Object/Map.h"
#include "Object/Object.h"

BottomUI::BottomUI(/*SystemMy* mystemMy*/) /*: _mystemMy(mystemMy)*/ {
    SetId("BottomUI");
}

void BottomUI::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);
}

void BottomUI::Update() {
    _y = Engine::Screen::height() - _height;
    _width = Engine::Screen::width();

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void BottomUI::Draw() {
    if (ImGui::Button("Reset##reset_btn", { 50.f, 20.f })) {
        Map::GetFirstCurrentMap().GetObjects().clear();
    }
}
