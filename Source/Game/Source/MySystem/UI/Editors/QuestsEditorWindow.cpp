//◦ Playrix ◦
#include "QuestsEditorWindow.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <Screen.h>
#include <FileManager.h>
#include "../../Quests/Quests.h"
#include "../../Quests/QuestManager.h"
#include "../../Commands/Functions.h"
#include <Object/Model.h>
#include "ImGuiManager/Editor/Common/CommonPopupModal.h"

namespace Editor {
    std::string QuestsEditorWindow::EditorCommand::emptyName = "EMPTY";

    void QuestsEditorWindow::OnOpen() {
        SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);

        float x = 225.f;
        float y = 100.f;

        ImGui::SetWindowPos(Id().c_str(), { x, y });
        ImGui::SetWindowSize(Id().c_str(), { _width, _height });

        UpdateSizes(_width, _height);
        LoadEditorDatas();
    }

    void QuestsEditorWindow::OnClose() {
        CommonPopupModal::Hide();
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

        CommonPopupModal::Draw();
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
            ImGui::SameLine(35.f);
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

            ImGui::Dummy(ImVec2(0.f, 10.f));
            if (ImGui::CollapsingHeader("Copmmons params")) {
                // Класс
                ImGui::Dummy(ImVec2(0.f, 0.f));
                ImGui::SameLine(35.f);
                ImGui::Text("Class");

                auto [indexListParam, listParam] = GetIndexOfListByName(QuestManager::GetClassName(_selectQuest), "QuestClass");
                ImGui::SameLine(100.f);
                ImGui::PushID(++_guiId);
                if (ImGui::Combo("", &indexListParam, listParam.data(), listParam.size())) {
                    //...
                }
                ImGui::PopID();

                // Описание
                ImGui::Dummy(ImVec2(0.f, 0.f));
                ImGui::SameLine(18.f);
                ImGui::Text("Description");

                ImGui::SameLine(100.f);
                ImGui::PushItemWidth(_widthQuest - 130.f);
                help::CopyToArrayChar(_textBuffer, _selectQuest->_description);
                ImGui::PushID(++_guiId);
                if (ImGui::InputTextMultiline("", _textBuffer.data(), _textBuffer.size(), { _widthQuest - 130.f, 35.f })) {
                    _selectQuest->_description = _textBuffer.data();
                }
                ImGui::PopID();
                ImGui::PopItemWidth();
            }

            // Команды
            ImGui::Dummy(ImVec2(0.f, 5.f));
            ImGui::PushItemWidth(_widthQuest / 2.1);

            DrawCommands(_selectQuest->_commandsOnInit,      observerLists.dataList[0]);
            DrawCommands(_selectQuest->_commandsOnTap,       observerLists.dataList[1]);
            DrawCommands(_selectQuest->_commandsOnUpdate,    observerLists.dataList[2]);
            DrawCommands(_selectQuest->_commandsOnCondition, observerLists.dataList[3]);
            DrawCommands(_selectQuest->_commandsDebug,       observerLists.dataList[4]);

            ImGui::EndChild();

            // Кнопки
            QuestButtonDisplay();
            
        ImGui::EndChild();
        ImGui::Dummy(ImVec2(0.f, 5.f));
    }

    void QuestsEditorWindow::DrawCommands(Commands& commands, const std::string& title) {
        if (commands.empty()) {
            return;
        }

        int countCommands = commands.size();
        std::function<void(void)> fun;

        if (ImGui::CollapsingHeader((title + " (" + std::to_string(countCommands) + ")").c_str())) {
            for (int index = 0; index < countCommands; ++index) {
                Command& command = commands[index];

                // Команда
                ImGui::BeginGroup();
                int comboCommandIndex = GetIndexOfList(command.id, _listCommands);
                ImGui::PushID(++_guiId);
                if (ImGui::Combo("", &comboCommandIndex, _listCommands.data(), _listCommands.size())) {
                    command.id = std::string(_listCommands[comboCommandIndex]);
                    command.parameters.clear();
                }
                ImGui::PopID();

                // Добавление параметра
                ImGui::Dummy(ImVec2(0.f, 0.f));
                ImGui::PushID(++_guiId);
                if (ImGui::Button("Add param", { 150.f, 20.f })) {
                    fun = [index, &commands]() {
                        auto removeIt = commands.begin() + index;
                        if (removeIt != commands.end()) {
                            removeIt->parameters.emplace_back();
                        }
                    };
                }
                ImGui::PopID();

                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Button, index == 0 ? Editor::disableColor : Editor::defaultColor);
                ImGui::PushID(++_guiId);
                if (ImGui::ArrowButton("", ImGuiDir_Up) && index != 0) {
                    fun = [index, &commands]() {
                        int prewIndex = index - 1;
                        if (prewIndex >= 0) {
                            std::swap(*(commands.begin() + index), *(commands.begin() + prewIndex));
                        }
                    };
                }
                ImGui::PopID();
                ImGui::PopStyleColor();

                // Кнопка удаления команды
                ImGui::Dummy(ImVec2(0.f, 0.f));
                ImGui::PushID(++_guiId);
                if (ImGui::Button("Remove command", { 150.f, 20.f })) {
                    fun = [index, &commands]() {
                        auto removeIt = commands.begin() + index;
                        if (removeIt != commands.end()) {
                            commands.erase(removeIt);
                        }
                    };
                }
                ImGui::PopID();

                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Button, index == (countCommands - 1) ? Editor::disableColor : Editor::defaultColor);
                ImGui::PushID(++_guiId);
                if (ImGui::ArrowButton("", ImGuiDir_Down) && index != (countCommands - 1)) {
                    fun = [index, &commands]() {
                        int prewIndex = index + 1;
                        if (prewIndex < commands.size()) {
                            std::swap(*(commands.begin() + index), *(commands.begin() + prewIndex));
                        }
                    };
                }
                ImGui::PopID();
                ImGui::PopStyleColor();

                //...................
                ImGui::PushID(++_guiId);
                ImGui::Checkbox("disable", &command.disable);
                ImGui::PopID();

                ImGui::EndGroup();

                // Параметры команды
                ImGui::SameLine();
                ImGui::BeginGroup();

                std::vector<std::string>& parameters = command.parameters;
                std::vector<std::string>& editorParams = GetEditorCommand(comboCommandIndex).params;
                int countParams = 0;

                if (editorParams.size() > parameters.size()) {
                    countParams = editorParams.size();
                    parameters.resize(countParams);
                } else if (parameters.size() > editorParams.size()) {
                    countParams = parameters.size();
                    editorParams.resize(countParams);
                }
                else {
                    countParams = editorParams.size();
                }

                for (size_t iParam = 0; iParam < countParams; ++iParam) {
                    std::string& parameter = parameters[iParam];
                    const std::string& editorParam = editorParams[iParam];

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
                ImGui::Dummy(ImVec2(0.f, 20.f));
                ImGui::Separator();
            }

            ImGui::Dummy(ImVec2(0.f, 5.f));
            ImGui::PushID(++_guiId);
            if (ImGui::Button("Add command.", { 200.f, 24.f })) {
                commands.emplace_back();
            }
            ImGui::PopID();

            /*ImGui::SameLine();
            if (ImGui::Button("Remove observer for event.", { 200.f, 24.f })) {
                //...
            }*/

            ImGui::Dummy(ImVec2(0.f, 20.f));
            ImGui::Separator();
        }

        if (fun) {
            fun();
        }
    }

    void QuestsEditorWindow::AddQuest() {
        Quest::Ptr newQuestPtr(new QuestStart("EMPTY")); // TODO:
        QuestManager::GetQuests().emplace_back(newQuestPtr);
    }

    void QuestsEditorWindow::CopyQuest(Quest::Ptr& questPtr) {
        if (questPtr) {
            //...
        }
    }
    
    void QuestsEditorWindow::RemoveQuest(Quest::Ptr& questPtr) {
        if (questPtr) {
            QuestManager::Remove(questPtr->Name());
        }
    }

    void QuestsEditorWindow::AddObserver() {
        Editor::CommonPopupModal::Show(GetSharedWndPtr(), [this, indexPtr = std::make_shared<int>(-1)]() {
            if (!_selectQuest) {
                CommonPopupModal::Hide();
                return;
            }

            ImGui::PushID(++_guiId);
            ImGui::Combo("", indexPtr.get(), observerLists.viewList.data(), observerLists.viewList.size());
            ImGui::PopID();

            Commands* commands = nullptr;

            switch (*indexPtr)
            {
            case 0: {
                commands = &_selectQuest->_commandsOnInit;
            } break;
            case 1: {
                commands = &_selectQuest->_commandsOnTap;
            } break;
            case 2: {
                commands = &_selectQuest->_commandsOnUpdate;
            } break;
            case 3: {
                commands = &_selectQuest->_commandsOnCondition;
            } break;
            case 4: {
                commands = &_selectQuest->_commandsDebug;
            } break;
            default:
                break;
            };

            ImGui::Dummy(ImVec2(0.f, 0.f));
            ImGui::PushStyleColor(ImGuiCol_Button, commands && commands->empty() ? Editor::greenColor : Editor::disableColor);
            if (ImGui::Button("Add##add_obs_btn", { 100.f, 26.f }) && commands && commands->empty()) {
                commands->emplace_back(EditorCommand::emptyName);
                CommonPopupModal::Hide();
            }
            ImGui::PopStyleColor();

            ImGui::SameLine();
            if (ImGui::Button("Close##close_obs_btn", { 100.f, 26.f })) {
                CommonPopupModal::Hide();
            }
        }, "Add observer.");
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

        ImGui::PushStyleColor(ImGuiCol_Button, _selectQuest ? Editor::defaultColor : Editor::disableColor);
        ImGui::SameLine();
        if (ImGui::Button("Copy", { buttonWidth, buttonHeight }) && _selectQuest) {
            CopyQuest(_selectQuest);
        }

        ImGui::SameLine();
        if (ImGui::Button("Remove", { buttonWidth, buttonHeight }) && _selectQuest) {
            RemoveQuest(_selectQuest);
            _selectQuest.reset();
        }
        ImGui::PopStyleColor();

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
            AddObserver();
        }

        ImGui::SameLine();
        if (ImGui::Button("Start this quest.", { buttonWidth, buttonHeight })) {
            CommandManager::Run(Command{ "StartQuest", { _selectQuest->Name() } });
            Close();
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
        // C:\Work\My\System\Source\Game\Source\MySystem\Quests/QuestManager.cpp
        // C:\Work\My\System\Source\Game\Source\MySystem\Commands\Functions/Actions.h

        EditorCommand& emptyEditorCommand = _editorCommands.emplace_back(EditorCommand::emptyName);

        EditorDatasParceFile("..\\..\\..\\Game\\Source\\MySystem\\Commands\\Functions.cpp");
        EditorDatasParceFile("..\\..\\..\\Game\\Source\\MySystem\\Quests\\QuestManager.cpp");
        EditorDatasParceFile("..\\..\\..\\Game\\Source\\MySystem\\Commands\\Functions\\Actions.h");

        //...
        if (int size = _editorCommands.size(); size > 0) {
            _listCommands.reserve(size);

            for (const EditorCommand& newEditorCommand : _editorCommands) {
                _listCommands.emplace_back(newEditorCommand.name.data());
            }
        }

        // TODO:
        /// QuestClass /QuestStart/Quest/QuestSphere100/QuestSphere
        std::vector<const char*>& listClass = _mapLists["QuestClass"];
        auto addChars = [&listClass](const std::string& text) {
            size_t len = text.length();
            char* chs = new char[len + 1];
            memcpy(chs, text.c_str(), len);
            chs[len] = '\0';
            listClass.emplace_back(chs);
        };
        for (const std::string& nameClass :QuestManager::GetListClasses()) {
            addChars(nameClass);
        }

        // Observers
        observerLists.Add("Commands on init");
        observerLists.Add("commands on tap");
        observerLists.Add("commands on update");
        observerLists.Add("commands on condition");
        observerLists.Add("commands debug");
        observerLists.MakeViewData();
    }

    void QuestsEditorWindow::EditorDatasParceFile(const std::string& filePathHame)
    {
        std::string fileText = Engine::FileManager::readTextFile(filePathHame);
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
                        std::string hashName = "/";
                        size_t lenParamWord = paramWord.size();
                        if (lenParamWord <= 10) {
                            hashName = paramWord;
                        }
                        else {
                            hashName += paramWord.substr(0, 4);
                            hashName += paramWord.substr((lenParamWord - 4), 4);
                            hashName += std::to_string(lenParamWord);
                        }

                        if (_mapLists.find(hashName) == _mapLists.end()) {
                            std::vector<std::string> paramWords = SeparateString(paramWord, "/");

                            std::vector<const char*>& listParams = _mapLists[hashName];
                            listParams.reserve(paramWords.size());

                            for (const std::string& pWord : paramWords) {
                                size_t len = pWord.length();
                                char* chs = new char[len + 1];
                                memcpy(chs, pWord.c_str(), len);
                                chs[len] = '\0';
                                listParams.emplace_back(chs);
                            }

                            newEditorCommand.params.emplace_back(hashName);
                        }
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
                        } else if (paramWord == "#QUEST") {
                            if (_mapLists.find(paramWord) == _mapLists.end()) {
                                std::vector<Quest::Ptr>& quests = QuestManager::GetQuests();
                                std::vector<const char*>& listParams = _mapLists[paramWord];
                                listParams.reserve(quests.size());

                                for (const Quest::Ptr& questPtr : quests) {
                                    const std::string& nameQuest = questPtr->Name();

                                    size_t len = nameQuest.length();
                                    char* chs = new char[len + 1];
                                    memcpy(chs, nameQuest.c_str(), len);
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
