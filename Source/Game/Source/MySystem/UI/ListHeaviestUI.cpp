
#include "ListHeaviestUI.h"
#include <cmath>
#include "imgui.h"
#include <glm/ext/scalar_constants.hpp>
#include "CommonData.h"
#include "Math/Vector.h"
#include "MySystem/MySystem.h"
#include "../Objects/Body.h"
#include "../Objects/Space.h"
#include "Core.h"
#include "Screen.h"
#include "Math/Vector.h"

ListHeaviestUI::ListHeaviestUI() : UI::Window(this) {
    SetId("ListHeaviestUI");
    Close();
}

ListHeaviestUI::ListHeaviestUI(MySystem* mySystem)
    : UI::Window(this)
    , _mySystem(mySystem)
{
    SetId("ListHeaviestUI");
}

void ListHeaviestUI::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize);

    _x = Engine::Screen::width() - _width - 10.f;

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });

    ++CommonData::lockAction;
}

void ListHeaviestUI::OnClose() {
    --CommonData::lockAction;
}

void ListHeaviestUI::Update() {

}

void ListHeaviestUI::Draw() {
    if (!_mySystem || !_mySystem->_space) {
        return;
    }

    int _guiId = 0;
    
    auto& heaviestInfo = _mySystem->_space->_heaviestInfo;

    for (auto& pair : heaviestInfo) {
        ImGui::PushID(++_guiId);
        if (ImGui::Button(pair.second.c_str(), { 90.f, 32.f })) {            
            _mySystem->_space->_focusBody = pair.first;
        }
        ImGui::PopID();

        ImGui::SameLine();

        ImGui::PushID(++_guiId);
        if (ImGui::Button("x", { 32.f, 32.f })) {
            _mySystem->_space->RemoveBody(pair.first);
        }
        ImGui::PopID();
    }
}
