// ◦ Xyz ◦
#include "QuestsEditorWindow.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <Screen.h>
#include <FileManager.h>
#include "../../Quests/Quests.h"
#include "../../Commands/Functions.h"
#include "../../Commands/Functions/QuestCondition.h"
#include <Object/Model.h>
#include <ImGuiManager/Editor/Common/CommonPopupModal.h>
#include "../../UI/CommonData.h"

namespace Editor
{
	std::string QuestsEditorWindow::EditorCommand::emptyName = "EMPTY";
	std::string QuestsEditorWindow::questClassesType = "#QuestClass#";
	std::string QuestsEditorWindow::observesType = "#OBSERVERS#";

	void QuestsEditorWindow::OnOpen()
	{
		CommonData::PushLockScreen();

		SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);

		float x = 225.f;
		float y = 100.f;

		ImGui::SetWindowPos(Id().c_str(), { x, y });
		ImGui::SetWindowSize(Id().c_str(), { _width, _height });

		UpdateSizes(_width, _height);
		LoadEditorDatas();

		const std::string& pathFileName = QuestManager::Instance().PathFileName();
		_questManager.Load(pathFileName);
	}

	void QuestsEditorWindow::OnClose()
	{
		CommonPopupModal::Hide();
		Clear();
		CommonData::PopLockScreen();
	}

	void QuestsEditorWindow::Draw()
	{
		// #STYLE
		ImGuiStyle& style = ImGui::GetStyle();
		style.FramePadding.y = 3;
		style.GrabMinSize = 12;
		style.GrabRounding = 0;
		style.FrameRounding = 0;
		style.WindowPadding = { 8.f, 8.f };

		_guiId = 0;

		DrawList();
		DrawQuest();
		ButtonDisplay();

		CommonPopupModal::Draw();
	}

	void QuestsEditorWindow::Update()
	{
		OnResize();
	}

	void QuestsEditorWindow::OnResize()
	{
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

	void QuestsEditorWindow::UpdateSizes(float width, float height)
	{
		_width = width;
		_height = height;

		volatile static float border = 26.f;
		_widthQuest = _width - _widthList - border;
		_topHeight = _height - _buttonsHeight;
		_buttonsWidth = _width - border;
	}

	void QuestsEditorWindow::DrawList()
	{
		volatile static float height = 200.f;
		volatile static float listButtonHeight = 48.f;

		ImGui::BeginChild("quest_list", { _widthList, _topHeight }, true);
		ImGui::BeginChild("list", { _widthList, (_topHeight - listButtonHeight) }, false);

		for (Quest::Ptr& questPtr : _questManager.GetQuests()) {
			ImGui::PushStyleColor(ImGuiCol_Button, _selectQuest == questPtr ? Editor::greenColor : Editor::defaultColor);

			ImGui::PushID(++_guiId);
			if (ImGui::Button(questPtr->Name().c_str(), { 180.f, 20.f })) {
				PrepareDraw(questPtr);
			}
			ImGui::PopID();

			ImGui::SameLine();
			ImGui::PushID(++_guiId);
			if (ImGui::ArrowButton("", ImGuiDir_Up)) {
				std::vector<Quest::Ptr>& quests = _questManager.GetQuests();

				if (quests.size() > 1) {
					auto itFirst = std::find_if(quests.rbegin(), quests.rend(), [questPtr](const Quest::Ptr& itQuestPtr) {
						return questPtr == itQuestPtr;
					});

					if (itFirst != quests.rend()) {
						auto itSecond = itFirst + 1;

						if (itSecond == quests.rend()) {
							itSecond = quests.rbegin();
						}

						std::swap(*itFirst, *itSecond);
					}
				}
			}
			ImGui::PopID();

			ImGui::SameLine();
			ImGui::PushID(++_guiId);
			if (ImGui::ArrowButton("", ImGuiDir_Down)) {
				std::vector<Quest::Ptr>& quests = _questManager.GetQuests();

				if (quests.size() > 1) {
					auto itFirst = std::find_if(quests.begin(), quests.end(), [questPtr](const Quest::Ptr& itQuestPtr) {
						return questPtr == itQuestPtr;
					});

					if (itFirst != quests.end()) {
						auto itSecond = itFirst + 1;

						if (itSecond == quests.end()) {
							itSecond = quests.begin();
						}

						std::swap(*itFirst, *itSecond);
					}
				}
			}
			ImGui::PopID();

			ImGui::PopStyleColor();
		}

		ImGui::EndChild();

		QuestListButtonDisplay();

		ImGui::EndChild();
	}

	void QuestsEditorWindow::DrawQuest()
	{
		ImGui::SameLine();

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
		if (ImGui::CollapsingHeader("Common params")) {
			// Класс
			ImGui::Dummy(ImVec2(0.f, 0.f));
			ImGui::SameLine(35.f);
			ImGui::Text("Class");

			EditorListT<std::string>& listParam = _mapLists[questClassesType];
			ImGui::SameLine(100.f);
			ImGui::PushID(++_guiId);
			if (ImGui::Combo("", &listParam.GetIndex(_questManager.GetClassName(_selectQuest)), listParam.viewList.data(), listParam.viewList.size())) {
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

			DrawQuestParams(_selectQuest->_params, "Params", _selectQuest->Name());
			DrawQuestParams(Quest::globalParams, "Global params");
		}

		// Команды
		ImGui::Dummy(ImVec2(0.f, 5.f));

		EditorListT<std::string>& observerLists = _mapLists[observesType];

		DrawCommands(_selectQuest->_commandsOnInit, observerLists.dataList[0]);
		DrawCommands(_selectQuest->_commandsOnTap, observerLists.dataList[1]);
		DrawCommands(_selectQuest->_commandsOnUpdate, observerLists.dataList[2]);
		DrawCommands(_selectQuest->_commandsOnCondition, observerLists.dataList[3]);
		DrawCommands(_selectQuest->_commandsDebug, observerLists.dataList[4]);

		ImGui::EndChild();

		// Кнопки
		QuestButtonDisplay();

		ImGui::EndChild();
		ImGui::Dummy(ImVec2(0.f, 5.f));
	}

	void QuestsEditorWindow::DrawQuestParams(std::map<std::string, std::string>& paramMap, const std::string& title, const std::string& questName)
	{
		std::function<void(void)> reloadParams = 0;

		if (ImGui::CollapsingHeader((title + " (" + std::to_string(paramMap.size()) + ")").c_str())) {
			for (std::pair<const std::string, std::string>& paramPair : paramMap) {
				ImGui::Text(paramPair.first.c_str());
				ImGui::SameLine();

				ImGui::SameLine(_widthQuest / 2.4f);
				ImGui::PushID(++_guiId);
				if (ImGui::Button(":", ImVec2(20.f, 20.f))) {
					ChangeParamDisplay(paramMap, paramPair.first, questName);
				}
				ImGui::PopID();

				ImGui::SameLine();
				ImGui::PushItemWidth(_widthQuest / 2.3f);
				ImGui::PushID(++_guiId);
				help::CopyToArrayChar(_textBuffer, paramPair.second);
				if (ImGui::InputText("", _textBuffer.data(), _textBuffer.size())) {
					paramPair.second = _textBuffer.data();
				}
				ImGui::PopID();
			}

			ImGui::Dummy(ImVec2(0.f, 0.f));
			ImGui::PushID(++_guiId);
			if (ImGui::Button("Add param", ImVec2(200.f, 20.f))) {
				ChangeParamDisplay(paramMap, "", questName);
			}
			ImGui::PopID();
		}
	}

	void QuestsEditorWindow::DrawParams(std::vector<std::string>& parameters, float level)
	{
		ImGui::BeginGroup();

		std::vector<std::string>& editorParams = _editorCommands.Get().params;
		int countParams = 0;

		if (editorParams.size() > parameters.size()) {
			countParams = editorParams.size();
			parameters.resize(countParams);
		}
		else if (parameters.size() > editorParams.size()) {
			countParams = parameters.size();
			editorParams.resize(countParams);
		}
		else {
			countParams = editorParams.size();
		}

		std::vector<std::string>* subCommandParam = nullptr;
		std::function<void()> additionView = 0;

		for (size_t iParam = 0; iParam < countParams; ++iParam) {
			std::string& parameter = parameters[iParam];
			const std::string& editorParam = editorParams[iParam];

			if (editorParam.empty()) {
				ImGui::PushID(++_guiId);
				help::CopyToArrayChar(_textBuffer, parameter);
				if (ImGui::InputText("", _textBuffer.data(), _textBuffer.size())) {
					parameter = _textBuffer.data();
				}
				ImGui::PopID();
			}
			else if (editorParam.front() == '!') { // "!COMMANDS" "!PARAMS"
				std::string editorQuestParam;
				std::string editorQuestName;

				if (int previParam = iParam - 1; previParam >= 0) {
					editorQuestName = parameters[previParam];
					editorQuestParam = editorParam + ':' + parameters[previParam];

					if (_mapLists.find(editorQuestParam) == _mapLists.end()) {
						editorQuestParam = editorParam;
						editorQuestName.clear();
					}
				}

				if (editorQuestParam == "!PARAMS:CUSTOMER") {
					ImGui::PushID(++_guiId);
					help::CopyToArrayChar(_textBuffer, parameter);
					if (ImGui::InputText("", _textBuffer.data(), _textBuffer.size())) {
						parameter = _textBuffer.data();
					}
					ImGui::PopID();
				}
				else {
					EditorListT<std::string>& listParam = _mapLists[editorQuestParam];
					ImGui::PushID(++_guiId);
					if (ImGui::Combo("", &listParam.GetIndex(parameter), listParam.viewList.data(), listParam.viewList.size())) {
						parameter = listParam.Get();
					}
					ImGui::PopID();

					if (editorParam == "!COMMANDS") {
						if (parameter.empty()) {
							additionView = [this, iParam, &parameters, &parameter, editorParam]() {
								ImGui::PushItemWidth(_widthQuest / 2.f - 130.f);
								ImGui::PushID(++_guiId);
								ImGui::InputText("", _newTextBuffer.data(), _newTextBuffer.size());
								ImGui::PopID();

								ImGui::SameLine();
								if (ImGui::Button("Add sub commands##add_sub_commandslabel_btn", ImVec2(120.f, 20.f)) && _newTextBuffer[0] != '\0') {
									int previParam = iParam - 1;
									std::string editorQuestName;
									if (previParam >= 0) {
										editorQuestName = parameters[previParam];
									}
									if (Quest::Ptr questPtr = _questManager.GetQuest(editorQuestName)) {
										parameter = _newTextBuffer.data();
										_newTextBuffer[0] = '\0';

										questPtr->_commandMap.emplace(parameter, Commands());

										std::string editorQuestParam = editorParam + ':' + editorQuestName;
										if (_mapLists.find(editorQuestParam) == _mapLists.end()) {
											_mapLists.emplace(editorQuestParam, EditorListT<std::string>());
										}
										EditorListT<std::string>& editListParam = _mapLists[editorQuestParam];
										editListParam.Add(parameter.data());
										editListParam.MakeViewData();
									}
								}
							};
						}
						else {
							additionView = [this, editorQuestName, parameter = listParam.Get(), newlevel = level + 1.f]() {
								if (Quest::Ptr questPtr = _questManager.GetQuest(editorQuestName)) {
									ImGui::SameLine();
									ImGui::BeginGroup();

									ImGui::Text("Sub commands");
									DrawCommands(questPtr->_commandMap[parameter], parameter, newlevel);
									ImGui::EndGroup();

									if (questPtr->_commandMap[parameter].empty()) {
										questPtr->_commandMap[parameter].emplace_back();
									}
								}
							};
						}
					}
				}
			}
			else {
				EditorListT<std::string>& listParam = _mapLists[editorParam];
				ImGui::PushID(++_guiId);
				if (ImGui::Combo("", &listParam.GetIndex(parameter), listParam.viewList.data(), listParam.viewList.size())) {
					parameter = listParam.Get();
				}
				ImGui::PopID();
			}
		}

		ImGui::EndGroup();
		ImGui::Dummy(ImVec2(0.f, 0.f));

		//...
		if (additionView) {
			additionView();
		}
	}

	void QuestsEditorWindow::DrawCommands(Commands& commands, const std::string& title, float level)
	{
		if (commands.empty()) {
			return;
		}

		ImGui::PushItemWidth((_widthQuest - (level * 10.f) - 30.f) / 2.f);

		int countCommands = commands.size();
		std::function<void(void)> fun;

		if (ImGui::CollapsingHeader((title + " (" + std::to_string(countCommands) + ")").c_str())) {
			for (int index = 0; index < countCommands; ++index) {
				Command& command = commands[index];

				// Команда
				ImGui::BeginGroup();

				ImGui::PushID(++_guiId);
				_editorCommands.GetIndex(command.id);
				if (ImGui::Combo("", &_editorCommands.currentIndex, _editorCommands.viewList.data(), _editorCommands.viewList.size())) {
					command.id = _editorCommands.Get().name;
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

				ImGui::SameLine();
				DrawParams(command.parameters, level);

				ImGui::Dummy(ImVec2(0.f, 20.f));
				ImGui::Separator();
			}

			ImGui::PushID(++_guiId);
			if (ImGui::Button("Add command.", { 200.f, 24.f })) {
				commands.emplace_back();
			}
			ImGui::PopID();
		}

		ImGui::PopItemWidth();

		if (fun) {
			fun();
		}
	}

	void QuestsEditorWindow::AddQuest()
	{
		Quest::Ptr newQuestPtr(new QuestStart("EMPTY")); // TODO:
		_questManager.GetQuests().emplace_back(newQuestPtr);
	}

	void QuestsEditorWindow::CopyQuest(Quest::Ptr& questPtr)
	{
		if (questPtr) {
			//...
		}
	}

	void QuestsEditorWindow::RemoveQuest(Quest::Ptr& questPtr)
	{
		if (questPtr) {
			_questManager.Remove(questPtr->Name());
		}
	}

	void QuestsEditorWindow::AddObserver()
	{
		Editor::CommonPopupModal::Show(
			GetSharedWndPtr(), [this]() {
				if (!_selectQuest) {
					CommonPopupModal::Hide();
					return;
				}

				EditorListT<std::string>& observerLists = _mapLists[observesType];

				ImGui::PushID(++_guiId);
				ImGui::Combo("", &observerLists.currentIndex, observerLists.viewList.data(), observerLists.viewList.size());
				ImGui::PopID();

				Commands* commands = nullptr;

				switch (observerLists.currentIndex) {
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
			},
			"Add observer.");
	}

	void QuestsEditorWindow::QuestListButtonDisplay()
	{
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

	void QuestsEditorWindow::ChangeParamDisplay(std::map<std::string, std::string>& paramMap, const std::string& name, const std::string& questName)
	{
		std::shared_ptr<std::string> namePtr = std::make_shared<std::string>(name);

		Editor::CommonPopupModal::Show(
			GetSharedWndPtr(), [this, &paramMap, questName, name, namePtr]() {
				ImGui::PushItemWidth(200.f);

				ImGui::PushID(++_guiId);
				help::CopyToArrayChar(_textBuffer, *namePtr);
				if (ImGui::InputText("", _textBuffer.data(), _textBuffer.size())) {
					*namePtr = _textBuffer.data();
				}
				ImGui::PopID();

				ImGui::Dummy(ImVec2(0.f, 0.f));
				bool enable = (name != *namePtr) || !namePtr->empty();
				ImGui::PushStyleColor(ImGuiCol_Button, enable ? Editor::greenColor : Editor::disableColor);
				if (ImGui::Button("Change##change_name_param_btn", { 100.f, 26.f }) && enable) {
					if (!name.empty()) {
						auto it = paramMap.find(name);
						if (it != paramMap.end()) {
							paramMap.emplace(*namePtr, it->second);
							paramMap.erase(it);
						}
					}
					else {
						paramMap.emplace(*namePtr, "");
					}

					std::string param = "!PARAMS";
					if (!questName.empty()) {
						param += (":" + questName);
					}

					auto itParam = _mapLists.find(param);

					if (itParam != _mapLists.end()) {
						EditorListT<std::string>& listParams = itParam->second;
						listParams.Clear();
						listParams.Reserve(paramMap.size());

						for (auto& paramPair : paramMap) {
							listParams.Add(paramPair.first.c_str());
							listParams.MakeViewData();
						}
					}

					CommonPopupModal::Hide();
				}
				ImGui::PopStyleColor();

				ImGui::SameLine();
				if (ImGui::Button("Close##close_name_param_btn", { 100.f, 26.f })) {
					CommonPopupModal::Hide();
				}
			},
			"Change name param.");
	}

	void QuestsEditorWindow::QuestButtonDisplay()
	{
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
			CommandManager::Run(Command { "StartQuest", { _selectQuest->Name() } });
			Close();
		}

		ImGui::EndChild();
	}

	void QuestsEditorWindow::ButtonDisplay()
	{
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
			_questManager.Reload();
		}

		//...
		ImGui::SameLine();
		if (ImGui::Button("Save", { buttonWidth, buttonHeight })) {
			_questManager.Save();
			QuestManager::Instance().Reload();
		}

		ImGui::SameLine();
		if (ImGui::Button("Close", { buttonWidth, buttonHeight })) {
			Close();
		}

		ImGui::EndChild();
	}

	void QuestsEditorWindow::PrepareDraw(Quest::Ptr& selectQuest)
	{
		if (!selectQuest) {
			return;
		}

		_selectQuest = selectQuest;
	}

	void QuestsEditorWindow::Clear()
	{
		_editorCommands.Clear();
		_mapLists.clear();
	}

	std::vector<std::string> SeparateString(const std::string& text, const std::string& separator)
	{
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

	void QuestsEditorWindow::LoadEditorDatas()
	{
		Clear();

		// C:\Work\My\System\Source\Resources\Files\System
		// C:\Work\My\System\Source\Game\Source\MySystem\Commands\Functions.h
		// C:\Work\My\System\Source\Game\Source\MySystem\Commands\Functions.cpp
		// C:\Work\My\System\Source\Game\Source\MySystem\Quests/QuestManager.cpp
		// C:\Work\My\System\Source\Game\Source\MySystem\Commands\Functions/Actions.h
		// C:\Work\My\System\Source\Game\Source\MySystem\Commands\Functions/QuestCondition.h

		EditorCommand& emptyEditorCommandT = _editorCommands.Add(EditorCommand::emptyName);

		EditorDatasParceFile("..\\..\\..\\Game\\Source\\MySystem\\Commands\\Functions.h");
		EditorDatasParceFile("..\\..\\..\\Game\\Source\\MySystem\\Commands\\Functions.cpp");
		EditorDatasParceFile("..\\..\\..\\Game\\Source\\MySystem\\Quests\\QuestManager.cpp");
		EditorDatasParceFile("..\\..\\..\\Game\\Source\\MySystem\\Commands\\Functions\\Actions.h");
		EditorDatasParceFile("..\\..\\..\\Game\\Source\\MySystem\\Commands\\Functions\\QuestCondition.h");

		_editorCommands.MakeViewData();

		EditorListT<std::string>& listClass = _mapLists[questClassesType];
		for (const std::string& nameClass : _questManager.GetListClasses()) {
			listClass.Add(nameClass.c_str());
		}

		// Observers
		EditorListT<std::string>& observerLists = _mapLists[observesType];
		observerLists.Add("Commands on init");
		observerLists.Add("commands on tap");
		observerLists.Add("commands on update");
		observerLists.Add("commands on condition");
		observerLists.Add("commands debug");

		for (auto& [first, second] : _mapLists) {
			second.MakeViewData();
		}
	}

	void QuestsEditorWindow::EditorDatasParceFile(const std::string& filePathHame)
	{
		std::string fileText = Engine::FileManager::readTextFile(filePathHame);
		if (fileText.empty()) {
			return;
		}

		size_t beginPos = 0;
		beginPos = fileText.find("///", beginPos); // /// = 3
		if (beginPos == fileText.npos) {
			return;
		}

		beginPos += 3;

		while (beginPos != fileText.npos) {
			size_t endPos = fileText.find('\n', beginPos);
			if (endPos == fileText.npos) {
				endPos = fileText.length() - 1;
			}

			std::string subText = fileText.substr(beginPos, (endPos - beginPos));
			PrepareCommand(SeparateString(subText, " "));

			beginPos = fileText.find("///", endPos);
			if (beginPos != fileText.npos) {
				beginPos += 3;
			}
		}
	}

	void QuestsEditorWindow::PrepareCommand(const std::vector<std::string>& words)
	{
		size_t size = words.size();
		if (size == 0) {
			return;
		}

		EditorCommand& newEditorCommandT = _editorCommands.Add(words.front());

		for (size_t index = 1; index < size; ++index) {
			const std::string& paramWord = words[index];

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
					EditorListT<std::string>& listParams = _mapLists[hashName];
					listParams.Reserve(paramWords.size());

					for (const std::string& pWord : paramWords) {
						listParams.Add(pWord.c_str());
					}
				}
				newEditorCommandT.params.emplace_back(hashName);
			}
			else if (paramWord.front() == '!') { // TODO: перенести общую механику в одну функцию
				if (paramWord == "!COMMANDS") {
					auto appendCommands = [this, &paramWord](const std::string& questName, const std::map<std::string, Commands>& comands) {
						if (comands.empty()) {
							return;
						}

						std::string nameCommands = paramWord;
						if (!questName.empty()) {
							nameCommands += ":" + questName;
						}

						if (_mapLists.find(nameCommands) == _mapLists.end()) {
							auto& listParams = _mapLists[nameCommands];
							listParams.Reserve(comands.size());

							for (auto& paramPair : comands) {
								listParams.Add(paramPair.first.c_str());
							}
						}
					};

					std::vector<Quest::Ptr>& quests = _questManager.GetQuests();

					for (const Quest::Ptr& questPtr : quests) {
						appendCommands(questPtr->Name(), questPtr->_commandMap);
					}
					appendCommands("", Quest::globalCommandsMap); // Глобальные команды
				}
				else if (paramWord == "!PARAMS") {
					auto appendParams = [this, &paramWord](const std::string& questName, const std::map<std::string, std::string>& params) {
						if (params.empty()) {
							return;
						}

						std::string nameParam = paramWord;
						if (!questName.empty()) {
							nameParam += ":" + questName;
						}

						if (_mapLists.find(nameParam) == _mapLists.end()) {
							auto& listParams = _mapLists[nameParam];
							listParams.Reserve(params.size());

							for (auto& paramPair : params) {
								listParams.Add(paramPair.first.c_str());
							}
						}
					};

					std::vector<Quest::Ptr>& quests = _questManager.GetQuests();

					for (const Quest::Ptr& questPtr : quests) {
						appendParams(questPtr->Name(), questPtr->_params);
					}
					appendParams("GLOBAL", Quest::globalParams);     // Глобальные переменные
					appendParams("GAME", quest::GetMapGameParams()); // Глобальные переменные
					appendParams("CUSTOMER", { { "", "" } });        // Глобальные переменные
				}

				newEditorCommandT.params.emplace_back(paramWord);
			}
			else if (paramWord.front() == '#') {
				if (paramWord == "#MODELS") {
					if (_mapLists.find(paramWord) == _mapLists.end()) {
						auto& listParams = _mapLists[paramWord];
						std::vector<std::string> listModels = Model::GetListModels();
						listParams.Reserve(listModels.size());

						for (const std::string& nameModel : listModels) {
							listParams.Add(nameModel.c_str());
						}
					}
				}
				else if (paramWord == "#QUESTS") {
					if (_mapLists.find(paramWord) == _mapLists.end()) {
						std::vector<Quest::Ptr>& quests = _questManager.GetQuests();
						auto& listParams = _mapLists[paramWord];
						listParams.Reserve(quests.size());

						listParams.Add("GLOBAL");
						listParams.Add("GAME");
						listParams.Add("CUSTOMER");

						for (const Quest::Ptr& questPtr : quests) {
							listParams.Add(questPtr->Name().c_str());
						}
					}
				}
				else if (paramWord == "#EXPRESSIONS") {
					if (_mapLists.find(paramWord) == _mapLists.end()) {
						std::vector<Quest::Ptr>& quests = _questManager.GetQuests();
						auto& listParams = _mapLists[paramWord];
						listParams.Add({ ">", ">=", "==", "!=", "<", "<=", "is_more", "is_more_or_equal", "is_equal", "is_not_equal", "is_less", "is_less_or_equal" });
					}
				}
				else if (paramWord == "#OPERATIONS") {
					if (_mapLists.find(paramWord) == _mapLists.end()) {
						std::vector<Quest::Ptr>& quests = _questManager.GetQuests();
						auto& listParams = _mapLists[paramWord];
						listParams.Add({ "+", "-", "*", "/" });
					}
				}

				newEditorCommandT.params.emplace_back(paramWord);
			}
			else {
				newEditorCommandT.params.emplace_back();
			}
		}
	}
}
