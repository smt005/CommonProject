
#include "ListHeaviestUI.h"
#include <cmath>
#include "imgui.h"
#include <glm/ext/scalar_constants.hpp>
#include "../Objects/SystemMap.h"
#include "../SystemMy.h"
#include "Math/Vector.h"
#include "../Objects/SystemClass.h"
#include "../Objects/SystemMap.h"
#include "../Objects/SystemMapArr.h"
#include "../Objects/SystemMapStaticArr.h"
#include "../Objects/SystemMapDouble.h"
#include "../Objects/SystemMapEasyMerger.h"
#include "../Objects/SystemMapShared.h"
#include "../SaveManager.h"
#include "Core.h"
#include "Screen.h"
#include "Math/Vector.h"

ListHeaviestUI::ListHeaviestUI() : UI::Window(this) {
    SetId("ListHeaviestUI");
    Close();
}

ListHeaviestUI::ListHeaviestUI(SystemMy* systemMy)
    : UI::Window(this)
    , _systemMy(systemMy)
{
    SetId("ListHeaviestUI");
}

void ListHeaviestUI::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize);

    _x = Engine::Screen::width() - _width - 10.f;

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void ListHeaviestUI::Update() {

}

void ListHeaviestUI::Draw() {
    if (!_systemMy || !_systemMy->_systemMap) {
        return;
    }
    
    auto& heaviestInfo = _systemMy->_systemMap->_heaviestInfo;

    for (auto& pair : heaviestInfo) {
        if (ImGui::Button(pair.second.c_str(), { 128.f, 32.f })) {            
            _systemMy->_systemMap->_focusBody = pair.first;
        }
    }
}
