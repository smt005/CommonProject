
#include "ListHeaviestUI.h"
#include <cmath>
#include "imgui.h"
#include <glm/ext/scalar_constants.hpp>
#include "CommonData.h"
#include "Math/Vector.h"
#include "MySystem/MySystem.h"
#include "../Objects/BodyData.h"
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
        ImGui::PushStyleColor(ImGuiCol_Button, pair.first == _mySystem->_space->_focusBody ? Editor::greenColor : Editor::defaultColor);

        if (ImGui::Button(pair.second.c_str(), { 110.f, 32.f })) {            
            _mySystem->_space->_focusBody = pair.first;
        }
        ImGui::PopID();
        ImGui::PopStyleColor();

        ImGui::SameLine();

        ImGui::PushID(++_guiId);
        if (ImGui::Button("x", { 32.f, 32.f })) {
            _mySystem->_space->RemoveBody(pair.first);
        }
        ImGui::PopID();
    }

    ImGui::Separator();
    ImGui::PushID(++_guiId);
    if (ImGui::Button("Detuch", { 150.f, 32.f })) {
        _mySystem->_space->_focusBody.reset();
    }
    ImGui::PopID();

    if (_mySystem->_space->_focusBody) {
        // Speed
        double speed = _mySystem->_space->_focusBody->Velocity().length();
        std::string speedStr = std::to_string(speed);

        if (speed != 0.0 && speedStr == "0.000000") {
            ImGui::Text("Speed: > 0.0");
        } else {
            ImGui::Text("Speed: %s", speedStr.c_str());
        }

        // Force
        float force = _mySystem->_space->_focusBody->Force().length();
        std::string forceStr = std::to_string(force);

        if (force != 0.0 && forceStr == "0.000000") {
            ImGui::Text("Forcr: > 0.0");
        } else {
            ImGui::Text("Forcr: %s", forceStr.c_str());
        }

        // Position
        auto pos = _mySystem->_space->_focusBody->GetPos();
        ImGui::Text("Pos: %s, %s, %s", std::to_string((int)round(pos.x)).c_str(), std::to_string((int)round(pos.y)).c_str(), std::to_string((int)round(pos.z)).c_str());

        // Scale
        float scale = _mySystem->_space->_focusBody->Scale();
        std::string scaleStr = std::to_string(scale);
        ImGui::Text("scale: %s", scaleStr.c_str());
    }
}
