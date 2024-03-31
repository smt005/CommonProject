// ◦ Xyz ◦
#include "CommandsWindow.h"
#include <imgui.h>
#include <Screen.h>
#include <FileManager.h>
#include "../../Commands/Commands.h"

CommandsWindow::CommandsWindow() : UI::Window(this) { }

void CommandsWindow::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize);

    //float x = Engine::Screen::width() / 2.f - _width / 2.f;
    //float y = Engine::Screen::height() / 2.f - _height / 2.f;

    float x = 20.f;
    float y = 30.f;

    ImGui::SetWindowPos(Id().c_str(), { x, y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });

    auto searchPath = Engine::FileManager::getResourcesDir() / "Commands";
    std::vector<std::string> commandFilePathNames;
    Engine::FileManager::FindFiles(searchPath, ".json", commandFilePathNames);

    for (const std::string& filePathNames : commandFilePathNames) {
        size_t lastPos = filePathNames.rfind('\\');
        size_t pointPos = filePathNames.rfind(".json");

        if (lastPos == filePathNames.npos) {
            continue;
        }
        ++lastPos;
        if (lastPos == filePathNames.npos) {
            continue;
        }
        if (pointPos == filePathNames.npos) {
            continue;
        }

        if (lastPos < pointPos) {
            std::string name = filePathNames.substr(lastPos, (pointPos - lastPos));
            _commands.emplace_back(name, filePathNames);
        }
    }
}

void CommandsWindow::Draw() {
    ImGui::Text("Test text.");

    ImGui::Separator();

    for (auto&[name, filePathName] : _commands) {
        if (ImGui::Button(name.c_str(), { 180.f, 32.f })) {
            CommandManager::Run(filePathName);
        }
    }

    ImGui::Separator();
    if (ImGui::Button("Close", { 180.f, 32.f})) {
        Close();
    }
}
