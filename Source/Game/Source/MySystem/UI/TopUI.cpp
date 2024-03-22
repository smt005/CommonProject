
#include "TopUI.h"
#include "imgui.h"

#include "Core.h"
#include "Screen.h"
#include "CommonData.h"
#include "../Objects/Space.h"
#include "MySystem/MySystem.h"
#include <cmath>
#include <glm/ext/scalar_constants.hpp>

TopUI::TopUI() : UI::Window(this) {
}

void TopUI::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground);
    SetAlpha(0.f);
}

void TopUI::OnClose() {
}

void TopUI::Update() {
    _width = Engine::Screen::width();

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void TopUI::Draw() {
    if (MySystem::currentSpace) {
        int time = 0;
        time = (int)MySystem::currentSpace->timePassed / 1000;

        if (time == 10) {
            minFPS = FPS;
            maxFPS = FPS;
        }
        if (time > 10) {
            minFPS = FPS < minFPS ? minFPS = FPS : minFPS;
            maxFPS = FPS > maxFPS ? maxFPS = FPS : maxFPS;
        }

        FPS = static_cast<int>(1 / Engine::Core::deltaTime());
        int countBody = MySystem::currentSpace->Objects().size();

        ImGui::Text("Time: %d Count: %d FPS: %d - %d - %d", time, countBody, minFPS, FPS, maxFPS);

        volatile static float offsetSettingBtn = 60.f;
        ImGui::SameLine(_width - offsetSettingBtn);
        if (ImGui::Button("[:]", { 50.f, 50.f })) {
            //...
        }
    }

    // Отобразить текст.
    {
        ImGuiIO& io = ImGui::GetIO();
        ImFontAtlas* atlas = io.Fonts;

        for (int i = 0; i < atlas->Fonts.Size; i++) {
            ImFont* font = atlas->Fonts.front();
            font->Scale = 3.f;

            ImGui::Dummy(ImVec2(0.f, 0.f));
            ImGui::PushFont(font);
            ImGui::TextColored({ 0.f, 0.f, 0.f, 1.f }, "%s", CommonData::textOnScreen.c_str());
            ImGui::PopFont();
            font->Scale = 1.f;
        }
    }
}
