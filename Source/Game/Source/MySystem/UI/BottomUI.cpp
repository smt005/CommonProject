// ◦ Xyz ◦

#include "BottomUI.h"
#include "imgui.h"
#include <string>
#include <functional>
#include "MainUI.h"
#include "Core.h"
#include "Screen.h"
#include "Common/Help.h"
#include "CommonData.h"
#include "MySystem/MySystem.h"
#include "../Objects/SpaceManager.h"
#include "Draw/DrawLight.h"
#include "Draw/Camera/CameraControlOutside.h"
#include "../Objects/Space.h"
#include "../Objects/BaseSpace.h"
#include "../Objects/SpaceGpuX0.h"

// AddObjectUI

AddObjectUI::AddObjectUI(const FunActions& funActionsArg)
    : UI::Window(this)
    , _funActions(funActionsArg)
{}

void AddObjectUI::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground);

    _y = Engine::Screen::height() - _y - _height;

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void AddObjectUI::Update() {
    //...
}

void AddObjectUI::Draw() {
    for (auto& pairAction :_funActions) {
        ImGui::Dummy(ImVec2(0.f, 0.f));
        ImGui::SameLine(8.f);

        if (ImGui::Button(pairAction.first.c_str(), { 50.f, 50.f })) {
            pairAction.second();

            if (BottomUI* bottomUI = dynamic_cast<BottomUI*>(UI::GetWindow<BottomUI>().get())) {
                bottomUI->_funAddObject = pairAction;
            }

            Close();
        }
    }
}

// SetViewUI

SetViewUI::SetViewUI(const FunActions& funActionsArg)
    : UI::Window(this)
    , _funActions(funActionsArg)
{}

void SetViewUI::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground);

    _x = Engine::Screen::width() - _width;
    _y = Engine::Screen::height() - _y - _height;

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void SetViewUI::Update() {
    //...
}

void SetViewUI::Draw() {
    for (auto& pairAction : _funActions) {
        ImGui::Dummy(ImVec2(0.f, 0.f));
        ImGui::SameLine(8.f);

        if (ImGui::Button(pairAction.first.c_str(), { 50.f, 50.f })) {
            pairAction.second();

            if (BottomUI* bottomUI = dynamic_cast<BottomUI*>(UI::GetWindow<BottomUI>().get())) {
                bottomUI->_funSetView = pairAction;
            }

            Close();
        }
    }
}

// BottomUI

void BottomUI::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground);

    _funAddObject.first = "...";
    _funAddObject.second = nullptr;
    _addBodyType = AddBodyType::NONE;

    _funSetView.first = "A##right_a";
    _funSetView.second = [this]() {
        _viewType = (ViewType)SpaceManager::SetView(_mySystem);
    };
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

    if (!MySystem::currentSpace) {
        return;
    }

    // #STYLE
    ImGuiStyle& style = ImGui::GetStyle();
    style.FramePadding.y = 18.f;
    style.GrabMinSize = 100;
    style.GrabRounding = 5;
    style.FrameRounding = 5;
    style.WindowPadding = { 1.f, 1.f };

    // LEFT
    ImGui::SameLine(8.f);

    if (ImGui::Button(_funAddObject.first.c_str(), { 50.f, 50.f }) && _funAddObject.second) {
        _funAddObject.second();
        _lockAddObject = true;

        UI::CloseWindowT<AddObjectUI>();
    }

    if (ImGui::IsItemHovered()) {
        if (!_lockAddObject) {
            GenerateFunAddObjectUI();
        }
    } else {
        _lockAddObject = false;
    }

    // CENTER
    ImGui::SameLine();

    volatile static float widthSpaceSlider = 140.f;
    float widthSlider = Engine::Screen::width() - widthSpaceSlider;
    widthSlider /= 2;

    int deltaTime = (int)MySystem::currentSpace->deltaTime;
    int countOfIteration = (int)MySystem::currentSpace->countOfIteration;

    ImGui::PushItemWidth(widthSlider);

    if (ImGui::SliderInt("##deltaTime_slider", &deltaTime, 1, 100, "error = %d")) {
        MySystem::currentSpace->deltaTime = (float)deltaTime;
    }

    ImGui::SameLine();

    static float _countOfIteration = 0.f;
    std::string text = "speed: " + std::to_string(countOfIteration);

    if (ImGui::SliderFloat("##time_speed_slider", &_countOfIteration, 0.8f, 8, text.c_str())) {
        countOfIteration = (int)std::expf(_countOfIteration - 1);
        MySystem::currentSpace->countOfIteration = countOfIteration;
    }

    ImGui::PopItemWidth();

    // RIGHT
    ImGui::SameLine();
   
    if (ImGui::Button(_funSetView.first.c_str(), { 50.f, 50.f }) && _funSetView.second) {
        _funSetView.second();
        _lockSetView = true;
        UI::CloseWindowT<SetViewUI>();
    }

    if (ImGui::IsItemHovered()) {
        if (!_lockSetView) {
            GenerateFunViewUI();
        }
    }
    else {
        _lockSetView = false;
    }

    // #STYLE
    style.FramePadding.y = 3.f;
    style.WindowPadding = { 8.f, 8.f };
}

void BottomUI::GenerateFunAddObjectUI() {
    if (UI::ShowingWindow<AddObjectUI>()) {
        return;
    }

    FunActions funActions;

    funActions.emplace_back("0##left_1", [this]() {
        _addBodyType = AddBodyType::ORBIT;
    });
    
    funActions.emplace_back("1##left_1", [this]() {
        _addBodyType = AddBodyType::NONE;
    });
    
    funActions.emplace_back("2##left_2", [this]() {
#ifdef _DEBUG
        int countBody = 1000;
#else
        int countBody = 3000;
#endif
    SpaceManager::AddObjects(MySystem::currentSpace.get(), countBody, 10000, -20);

    });
    
    funActions.emplace_back("3##left_3", [this]() {
        _addBodyType = AddBodyType::NONE;
    });

    UI::ShowWindow<AddObjectUI>(funActions);
}

void BottomUI::GenerateFunViewUI() {
    if (UI::ShowingWindow<SetViewUI>()) {
        return;
    }

    FunActions funActions;

    funActions.emplace_back("A##right_a", [this]() {
        _viewType = (ViewType)SpaceManager::SetView(_mySystem);
    });

    funActions.emplace_back("B##right_b", []() {
        help::log("B");
    });

    funActions.emplace_back("C##right_c", []() {
        help::log("C");
    });

    funActions.emplace_back("D##right_d", []() {
        help::log("D");
    });

    UI::ShowWindow<SetViewUI>(funActions);
}
