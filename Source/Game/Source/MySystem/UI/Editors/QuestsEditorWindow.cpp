#include "QuestsEditorWindow.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <Screen.h>
#include <FileManager.h>
#include "../../Quests/QuestManager.h"
#include "../../Commands/Functions.h"
#include <Object/Model.h>

namespace Editor {
    void QuestsEditorWindow::OnOpen() {
        SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);

        float x = 225.f;
        float y = 30.f;

        ImGui::SetWindowPos(Id().c_str(), { x, y });
        ImGui::SetWindowSize(Id().c_str(), { _width, _height });

        UpdateSizes(_width, _height);
        LoadEditorDatas();
    }

    void QuestsEditorWindow::OnClose() {
        Clear();
    }

    void QuestsEditorWindow::Draw() {
        // #STYLE
        ImGuiStyle& style = ImGui::GetStyle();
        style.FramePadding.y = 3;
        style.GrabMinSize = 12;
        style.GrabRounding = 0;
        style.FrameRounding = 0;
        style.WindowPadding = { 8.f, 8.f };

        _guiId = 0;

        DrawList();

        ImGui::SameLine();
        DrawQuest();

        ButtonDisplay();
    }

    void QuestsEditorWindow::Update() {
        OnResize();
    }

    void QuestsEditorWindow::OnResize() {
        ImGuiWindow* imGuiWindow = ImGui::FindWindowByName(Id().c_str());
        if (!imGuiWindow) {
            return;
        }

        float width = imGuiWindow->Size.x;
        float height = imGuiWindow->Size.y;

        if (_width != width || _height != height) {
            UpdateSizes(width, height);
        }
    }

    void QuestsEditorWindow::UpdateSizes(float width, float height) {
        _width = width;
        _height = height;

        volatile static float border = 26.f;
        _widthQuest = _width - _widthList - border;
        _topHeight = _height - _buttonsHeight;
        _buttonsWidth = _width - border;
    }

    void QuestsEditorWindow::DrawList() {
        volatile static float height = 200.f;
        volatile static float listButtonHeight = 48.f;

        ImGui::BeginChild("quest_list", { _widthList, _topHeight }, true);
            ImGui::BeginChild("list", { _widthList, (_topHeight - listButtonHeight) }, false);

            for (Quest::Ptr& questPtr : QuestManager::GetQuests()) {
                ImGui::PushStyleColor(ImGuiCol_Button, _selectQuest == questPtr ? Editor::greenColor : Editor::defaultColor);

                ImGui::PushID(++_guiId);
                if (ImGui::Button(questPtr->Name().c_str(), { 180.f, 20.f })) {
                    PrepareDraw(questPtr);
                }
                ImGui::PopID();
            
                ImGui::SameLine();
                ImGui::PushID(++_guiId);
                if (ImGui::ArrowButton("", ImGuiDir_Up)) {
                    //...
                }
                ImGui::PopID();

                ImGui::SameLine();
                ImGui::PushID(++_guiId);
                if (ImGui::ArrowButton("", ImGuiDir_Down)) {
                    //...
                }
                ImGui::PopID();

                ImGui::PopStyleColor();
            }

            ImGui::EndChild();

            QuestListButtonDisplay();
        
        ImGui::EndChild();
    }

    void QuestsEditorWindow::DrawQuest() {
        if (!_selectQuest) {
            ImGui::BeginChild("quest", { _widthQuest, _topHeight }, true);
            ImGui::EndChild();
            return;
        }

        ImGui::BeginChild("quest", { _widthQuest, _topHeight }, true);
            ImGui::BeginChild("quest_com", { _widthQuest, _topHeight - 48.f }, false);
            // Параметры квеста
            ImGui::SameLine(30.f);
            ImGui::Text("Name");

            ImGui::SameLine(100.f);
            ImGui::PushItemWidth(_widthQuest - 130.f);
            help::CopyToArrayChar(_textBuffer, _selectQuest->Name());
            ImGui::PushID(++_guiId);
            if (ImGui::InputText("", _textBuffer.data(), _textBuffer.size())) {
                _selectQuest->_name = _textBuffer.data();
            }
            ImGui::PopID();
            ImGui::PopItemWidth();

            // Команды
            ImGui::Dummy(ImVec2(0.f, 10.f));
            ImGui::PushItemWidth(_widthQuest / 2.1);

            DrawCommands(_selectQuest->_commands,            "commands on init");
            DrawCommands(_selectQuest->_commandsOnTap,       "commands on tap");
            DrawCommands(_selectQuest->_commandsOnCondition, "commands on condition");

            ImGui::EndChild();

            // Кнопки
            QuestButtonDisplay();

        ImGui::EndChild();
    }

    void QuestsEditorWindow::DrawCommands(Commands& commands, const std::string& title) {
        if (commands.empty()) {
            return;
        }

        int removeIndex = -1;

        if (ImGui::CollapsingHeader(title.c_str())) {
            for (int index = 0; index < commands.size(); ++index) {
                Command& command = commands[index];

                // Команда
                ImGui::BeginGroup();
                int comboCommandIndex = GetIndexOfList(command.id, _listCommands);
                ImGui::PushID(++_guiId);
                if (ImGui::Combo("", &comboCommandIndex, _listCommands.data(), _listCommands.size())) {
                    command.id = std::string(_listCommands[comboCommandIndex]);
                }
                ImGui::PopID();

                // Кнопка удаления команды
                ImGui::Dummy(ImVec2(0.f, 0.f));
                ImGui::PushID(++_guiId);
                if (ImGui::Button("Remove commands", { 150.f, 24.f })) {
                    removeIndex = index;
                }
                ImGui::PopID();

                ImGui::EndGroup();

                // Параметры команды
                ImGui::SameLine();
                ImGui::BeginGroup();

                std::vector<std::string>& parameters = command.parameters;
                std::vector<std::string>& editorParams = GetEditorCommand(comboCommandIndex).params;

                int countParams = parameters.size();
                if (editorParams.size() < countParams) {
                    editorParams.resize(countParams);
                }

                for (size_t iParam = 0; iParam < countParams; ++iParam) {
                    std::string& parameter = parameters[iParam];
                    const std::string& editorParam = editorParams[iParam];

                    ImGui::Dummy(ImVec2(0.f, 0.f));
                    if (editorParams[iParam].empty()) {
                        ImGui::PushID(++_guiId);
                        help::CopyToArrayChar(_textBuffer, parameter);
                        if (ImGui::InputText("", _textBuffer.data(), _textBuffer.size())) {
                            parameter = _textBuffer.data();
                        }
                        ImGui::PopID();
                    }
                    else {
                        auto [indexListParam, listParam] = GetIndexOfListByName(parameter, editorParam);
                        ImGui::PushID(++_guiId);
                        if (ImGui::Combo("", &indexListParam, listParam.data(), listParam.size())) {
                            parameter = std::string(listParam[indexListParam]);
                        }
                        ImGui::PopID();
                    }
                }

                ImGui::EndGroup();
                ImGui::Separator();
            }

            ImGui::Dummy(ImVec2(0.f, 5.f));
            if (ImGui::Button("Remove observer for event.", { 200.f, 24.f })) {
                //...
            }
            ImGui::Dummy(ImVec2(0.f, 10.f));
            ImGui::Separator();
        }

        if (removeIndex != -1) {
            auto removeIt = commands.begin() + removeIndex;

            if (removeIt != commands.end()) {
                commands.erase(removeIt);
                Clear();
            }
        }
    }

    void QuestsEditorWindow::AddQuest() {
        Quest::Ptr newQuestPtr(new Quest("EMPTY"));
        QuestManager::GetQuests().emplace_back(newQuestPtr);
    }

    void QuestsEditorWindow::CopyQuest(Quest::Ptr& questPtr) {
        if (questPtr) {
            Quest::Ptr newQuestPtr(new Quest(*questPtr));
            newQuestPtr->_name = questPtr->Name() + "_copy";
            QuestManager::GetQuests().emplace_back(newQuestPtr);
        }
    }
    
    void QuestsEditorWindow::RemoveQuest(Quest::Ptr& questPtr) {
        if (questPtr) {
            QuestManager::Remove(questPtr->Name());
        }
    }

    void QuestsEditorWindow::QuestListButtonDisplay() {
        volatile static float listButtonWidth = 0;
        volatile static float listButtonHeight = 28.f;
        volatile static float buttonWidth = 77;
        volatile static float buttonHeight = 24.f;
        volatile static float buttonOffsetHeight = 8.f;

        ImGui::BeginChild("list_buttons", { _widthList, listButtonHeight }, false);
        ImGui::Separator();

        //...
        if (ImGui::Button("Add", { buttonWidth, buttonHeight })) {
            AddQuest();
        }

        ImGui::SameLine();
        if (ImGui::Button("Copy", { buttonWidth, buttonHeight })) {
            CopyQuest(_selectQuest);
        }

        ImGui::SameLine();
        if (ImGui::Button("Remove", { buttonWidth, buttonHeight })) {
            RemoveQuest(_selectQuest);
        }

        ImGui::EndChild();
    }

    void QuestsEditorWindow::QuestButtonDisplay() {
        volatile static float listButtonWidth = 0;
        volatile static float listButtonHeight = 28.f;
        volatile static float buttonWidth = 200.f;
        volatile static float buttonHeight = 24.f;
        volatile static float buttonOffsetHeight = 8.f;

        ImGui::BeginChild("list_buttons_q", { 400.f, listButtonHeight }, false);
        ImGui::Separator();

        //...
        if (ImGui::Button("Add observer of event", { buttonWidth, buttonHeight })) {
            //AddQuest();
        }

        ImGui::EndChild();
    }

    void QuestsEditorWindow::ButtonDisplay() {
        volatile static float buttonWidth = 100.f;
        volatile static float buttonHeight = 32.f;
        volatile static float buttonOffsetHeight = 8.f;
        volatile static float listButtonHeight = 36.f;
        volatile static float offsetButton = 336.f;

        ImGui::BeginChild("list_buttons", { (_width - 0.f), listButtonHeight }, false);
        ImGui::SameLine((_width - offsetButton));

        if (ImGui::Button("Reset", { buttonWidth, buttonHeight })) {
            _selectQuest.reset();
            Clear();
            QuestManager::Reload();
        }

        //...
        ImGui::SameLine();
        if (ImGui::Button("Save", { buttonWidth, buttonHeight })) {
            QuestManager::Save();
        }

        ImGui::SameLine();
        if (ImGui::Button("Close", { buttonWidth, buttonHeight })) {
            Close();
        }

        ImGui::EndChild();
    }

    void QuestsEditorWindow::PrepareDraw(Quest::Ptr& selectQuest) {
        if (!selectQuest) {
            return;
        }

        _selectQuest = selectQuest;
    }

    void QuestsEditorWindow::Clear() {
        //_listCommands, данные хранятся в _editorCommands

        for (const auto& pairList : _mapLists) {
            for (const char* chars : pairList.second) {
                delete[] chars;
            }
        }
        _mapLists.clear();
    }

    std::vector<std::string> SeparateString(const std::string& text, const std::string& separator) {
        std::vector<std::string> strings;
        if (text.empty()) {
            return strings;
        }

        size_t beginPos = 0;
        size_t endPos = 0;

        while (beginPos != text.npos) {
            endPos = text.find(separator, beginPos);
            if (beginPos == endPos) {
                ++beginPos;
                continue;
            }

            if (endPos == text.npos) {
                endPos = text.length();
                if (beginPos != endPos) {
                    strings.emplace_back(text.substr(beginPos, (endPos - beginPos)));
                }
                break;
            }

            strings.emplace_back(text.substr(beginPos, (endPos - beginPos)));
            beginPos = endPos;
        }
        
        return strings;
    }

    void QuestsEditorWindow::LoadEditorDatas() {
        // C:\Work\My\System\Source\Resources\Files\System
        // C:\Work\My\System\Source\Game\Source\MySystem\Commands\Functions.cpp
        std::string fileText = Engine::FileManager::readTextFile("..\\..\\..\\Game\\Source\\MySystem\\Commands\\Functions.cpp");
        if (fileText.empty()) {
            return;
        }
        else {
            help::Log("OK");
        }

        size_t beginPos = 0;
        beginPos = fileText.find("///", beginPos); // /// = 3
        beginPos += 3;

        while (beginPos != fileText.npos) {
            size_t endPos = fileText.find('\n', beginPos);
            if (endPos == fileText.npos) {
                endPos = fileText.length() - 1;
            }

            std::string subText = fileText.substr(beginPos, (endPos - beginPos));
            std::vector<std::string> words = SeparateString(subText, " ");
            //.............................................................................
            size_t size = words.size();
            if (size > 0) {
                EditorCommand& newEditorCommand = _editorCommands.emplace_back();

                //...
                newEditorCommand.name = words.front();

                //...
                for (size_t index = 1; index < size; ++index) {
                    const std::string& paramWord = words[index];

                    // Список
                    if (paramWord.front() == '/') {
                        std::vector<std::string> paramWords = SeparateString(paramWord, "/");
                        std::string nameParam = newEditorCommand.name + std::to_string(index);

                        std::vector<const char*>& listParams =_mapLists[nameParam];
                        listParams.reserve(paramWords.size());

                        for (const std::string& pWord :paramWords) {
                            size_t len = pWord.length();
                            char* chs = new char[len + 1];
                            memcpy(chs, pWord.c_str(), len);
                            chs[len] = '\0';
                            listParams.emplace_back(chs);
                        }

                        newEditorCommand.params.emplace_back(nameParam);
                    }
                    else if (paramWord.front() == '#') {
                        if (paramWord == "#MODEL") {
                            if (_mapLists.find(paramWord) == _mapLists.end()) {
                                std::vector<const char*>& listParams = _mapLists[paramWord];
                                std::vector<std::string> listModels = Model::GetListModels();
                                listParams.reserve(listModels.size());

                                for (const std::string& nameModel : listModels) {
                                    size_t len = nameModel.length();
                                    char* chs = new char[len + 1];
                                    memcpy(chs, nameModel.c_str(), len);
                                    chs[len] = '\0';
                                    listParams.emplace_back(chs);
                                }
                            }

                            newEditorCommand.params.emplace_back(paramWord);
                        }
                    }
                    else {
                        newEditorCommand.params.emplace_back();
                    }
                }
            }
            //.............................................................................

            beginPos = fileText.find("///", endPos);
            if (beginPos != fileText.npos) {
                beginPos += 3;
            }
        }

        //...
        if (int size = _editorCommands.size(); size > 0) {
            _listCommands.reserve(size);

            for (const EditorCommand& newEditorCommand : _editorCommands) {
                _listCommands.emplace_back(newEditorCommand.name.data());
            }
        }
    }

    std::pair<int, const std::vector<const char*>&> QuestsEditorWindow::GetIndexOfListByName(const std::string& text, const std::string& nameList) {
        auto it = _mapLists.find(nameList);
        if (it == _mapLists.end()) {
            return { -1, std::vector<const char*>() };
        }
        return { GetIndexOfList(text, it->second), it ->second};
    }

    int QuestsEditorWindow::GetIndexOfList(const std::string& text, const std::vector<const char*>& listTexts)
    {
        int index = -1;

        for (const char* chars : listTexts) {
            ++index;

            if (text == chars) {
                return index;
            }
        }

        return -1;
    }

    QuestsEditorWindow::EditorCommand& QuestsEditorWindow::GetEditorCommand(int index)
    {
        if (index >= 0 && index < _editorCommands.size()) {
            return _editorCommands[index];
        }

        auto itDefault = std::find_if(_editorCommands.begin(), _editorCommands.end(), [](const EditorCommand& command) {
            return command.name == "DEFAULT";
        });
        
        return itDefault != _editorCommands.end() ? *itDefault : _editorCommands.emplace_back("DEFAULT");
    }
}
