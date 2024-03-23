// ◦ Xyz ◦
#include "QuestsWindow.h"
#include <imgui.h>
#include <Screen.h>
#include <FileManager.h>
#include <MyStl/Event.h>
#include "../../Commands/Commands.h"
#include "../../Quests/QuestManager.h"

QuestsWindow::QuestsWindow() : UI::Window(this) { }

void QuestsWindow::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize);

    //float x = Engine::Screen::width() / 2.f - _width / 2.f;
    //float y = Engine::Screen::height() / 2.f - _height / 2.f;

    float x = 225.f;
    float y = 30.f;

    ImGui::SetWindowPos(Id().c_str(), { x, y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
}

void QuestsWindow::Draw() {
    ImGui::Text("Test text.");

    ImGui::Separator();

    for (const Quest::Ptr& questPtr : QuestManager::GetQuests()) {
        if (ImGui::Button(questPtr->Name().c_str(), { 180.f, 32.f })) {
            for (const Quest::Ptr& questPtr : QuestManager::GetQuests()) {
                Event::Instance().Remove(questPtr->Name());
            }
            CommandManager::Run(Command("SetActiveQuest", { questPtr->Name(), "ACTIVE" }));
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Close", { 180.f, 32.f})) {
        Close();
    }
}
