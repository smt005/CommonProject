
#include "BottomUI.h"
#include "imgui.h"
#include <string>
#include "Core.h"
#include "Screen.h"
#include "CommonData.h"
#include "../SystemMy.h"
#include "Draw/DrawLight.h"
#include "Draw/Camera/CameraControlOutside.h"
#include "../Objects/SystemClass.h"
#include "../Objects/SystemMapEasyMerger.h"
#include "../Objects/SystemMapShared.h"
#include "../Objects/SystemMapMyShared.h"

BottomUI::BottomUI() : UI::Window(this) {
    Close();
}

BottomUI::BottomUI(SystemMy* systemMy)
    : UI::Window(this)
    , _systemMy(systemMy)
{}

void BottomUI::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove);
    SetAlpha(0.f);
}

void BottomUI::OnClose() {
}

void BottomUI::Update() {
    _y = Engine::Screen::height() - _height;
    _width = Engine::Screen::width();

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void BottomUI::Draw() {
    volatile static float framePadding = 18.f;
    volatile static float offsetBottom = 90.f;

    ImGuiStyle& style = ImGui::GetStyle();
    style.FramePadding.y = 18.f;
    style.GrabMinSize = 100;
    style.FrameRounding = 3;

    // Bottom
    ImGui::Dummy(ImVec2(0.f, (_height - offsetBottom)));

    bool orbite = false;

    if (orbite == 0) {
        if (ImGui::Button("[O]", { 50.f, 50.f })) {
            //...
        }
    }
    else {
        if (ImGui::Button("[S]", { 50.f, 50.f })) {
            //...
        }
    }

    //...
    ImGui::SameLine();

#if  SYSTEM_MAP < 8
    volatile static float widthSlider = 190.f;

    ImGui::PushItemWidth((Engine::Screen::width() - widthSlider));
    ImGui::SliderInt("##time_speed_slider", &timeSpeed, 0, 110);
    _systemMy->_timeSpeed = timeSpeed;
    ImGui::PopItemWidth();
#else
    volatile static float widthSpaceSlider = 140.f;
    float widthSlider = Engine::Screen::width() - widthSpaceSlider;
    widthSlider /= 2;

    int deltaTime = (int)_systemMy->_systemMap->deltaTime;
    int countOfIteration = (int)_systemMy->_systemMap->countOfIteration;

    ImGui::PushItemWidth(widthSlider);

    if (ImGui::SliderInt("##deltaTime_slider", &deltaTime, 1, 100, "error = %d")) {
        _systemMy->_systemMap->deltaTime = (double)deltaTime;
    }

    ImGui::SameLine();
    if (ImGui::SliderInt("##time_speed_slider", &countOfIteration, 0, 1000, "speed = %d")) {
        _systemMy->_systemMap->countOfIteration = countOfIteration;
    }

    ImGui::PopItemWidth();
#endif //  SYSTEM_MAP < 8

    //...
    ImGui::SameLine();
    /*if (_systemMy->ViewByObject()) {
        if (ImGui::Button("[X]", { 50.f, 50.f })) {
            _systemMy->SetViewByObject(false);
        }
    }
    else {
        if (ImGui::Button("[V]", { 50.f, 50.f })) {
            _systemMy->SetViewByObject(true);
        }
    }*/

    //...
    ImGui::SameLine();
    /*if (_systemMy->PerspectiveView()) {
        if (ImGui::Button("[*]", { 50.f, 50.f })) {
            _systemMy->SetPerspectiveView(false);
        }
    }
    else {
        if (ImGui::Button("[#]", { 50.f, 50.f })) {
            _systemMy->SetPerspectiveView(true);
        }
    }*/


    if (ImGui::Button((_systemMy->_systemMap->threadEnable ? "[V]" : "[#]"), { 50.f, 50.f })) {
         if (_systemMy->_camearCurrent == _systemMy->_camearSide) {
             _systemMy->_camearCurrent = _systemMy->_camearTop;
             DrawLight::setClearColor(0.7f, 0.8f, 0.9f, 1.0f); 
             
         }
         else if (_systemMy->_camearCurrent == _systemMy->_camearTop) {
             _systemMy->_camearCurrent = _systemMy->_camearSide;
             DrawLight::setClearColor(0.1f, 0.2f, 0.3f, 1.0f);
         }
    }

    //...
    style.FramePadding.y = 3.f;
}
