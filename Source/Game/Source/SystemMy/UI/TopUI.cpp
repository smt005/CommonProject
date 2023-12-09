
#include "TopUI.h"
#include "imgui.h"

#include "Core.h"
#include "Screen.h"
#include "../SystemMy.h"

#include "../Objects/SystemClass.h"
#include "../Objects/SystemMap.h"
#include "../Objects/SystemMapArr.h"
#include "../Objects/SystemMapStaticArr.h"
#include "../Objects/SystemMapMyVec.h"
#include "../Objects/SystemMapDouble.h"
#include "../Objects/SystemMapEasyMerger.h"

#include <cmath>
#include <glm/ext/scalar_constants.hpp>

TopUI::TopUI() : UI::Window(this) {
    SetId("TopUI");
    Close();
}

TopUI::TopUI(SystemMy* systemMy)
    : UI::Window(this)
    , _systemMy(systemMy)
{
    SetId("TopUI");
}

void TopUI::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove);
    SetAlpha(0.f);
}

void TopUI::Update() {
    _width = Engine::Screen::width();

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void TopUI::Draw() {
    if (_systemMy && _systemMy->_systemMap) {
        int time = _systemMy->_systemMap->time;
        if (time == 10) {
            minFPS = FPS;
            maxFPS = FPS;
        }
        if (time > 10) {
            minFPS = FPS < minFPS ? minFPS = FPS : minFPS;
            maxFPS = FPS > maxFPS ? maxFPS = FPS : maxFPS;
        }

        FPS = static_cast<int>(1 / Engine::Core::deltaTime());
        int countBody = _systemMy->_systemMap->Objects().size();

        ImGui::Text("Time: %d Count: %d FPS: %d - %d - %d", time, countBody, minFPS, FPS, maxFPS);

        volatile static float offsetSettingBtn = 60.f;
        ImGui::SameLine(_width - offsetSettingBtn);
        if (ImGui::Button("[:]", { 50.f, 50.f })) {
            _systemMy->SetViewByObject(false);
        }
    }
}
