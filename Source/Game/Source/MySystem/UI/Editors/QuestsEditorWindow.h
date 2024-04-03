// ◦ Xyz ◦
#pragma once

#include "ImGuiManager/UI.h"
#include <memory>
#include <string>
#include <vector>
#include "../../Quests/Quest.h"
#include "../../Commands/Commands.h"
#include "Common/Help.h"
#include "ImGuiManager/Editor/Common/CommonUI.h"

namespace Editor {
	class QuestsEditorWindow final : public UI::Window {
	private:
		struct EditorCommand {
			std::string name;
			std::vector<std::string> params;
			EditorCommand() = default;
			EditorCommand(const std::string& _name) :name(_name) {}
		};

		struct EditorList {
			std::vector<std::string> dataList;
			std::vector<const char*> viewList;

			void Add(const std::string& text) {
				std::string newText = dataList.emplace_back(text);
			}
			void MakeViewData() {
				viewList.clear();
				for (std::string& text : dataList) {
					viewList.emplace_back(text.data());
				}
			}
		};

	public:
		QuestsEditorWindow() : UI::Window(this) {};
		void OnOpen() override;
		void OnClose() override;
		void Draw() override;
		void Update() override;

		void OnResize();
		void UpdateSizes(float width, float height);

		void DrawList();
		void DrawQuest();
		void QuestButtonDisplay();
		void QuestListButtonDisplay();

		void DrawCommands(Commands& commands, const std::string& title);
		void ButtonDisplay();
		void PrepareDraw(Quest::Ptr& selectQuest);

		void AddQuest();
		void CopyQuest(Quest::Ptr& questPtr);
		void RemoveQuest(Quest::Ptr& questPtr);
		void AddObserver();
		void Clear();

		void LoadEditorDatas();
		void EditorDatasParceFile(const std::string& filePathHame);

		std::pair<int, const std::vector<const char*>&> GetIndexOfListByName(const std::string& text, const std::string& nameList);
		int GetIndexOfList(const std::string& text, const std::vector<const char*>& listTexts);
		EditorCommand& GetEditorCommand(int index);

		inline std::shared_ptr<bool> GetSharedWndPtr() {
			if (!_sharedWndPtr) {
				_sharedWndPtr = std::make_shared<bool>(true);
			}
			return _sharedWndPtr;
		}

	private:
		float _width = 1000.f;
		float _height = 500.f;
		float _widthList = 263.f;
		float _widthQuest = 500.f;
		float _topHeight = 100.f;
		float _buttonsWidth = 200.f;
		float _buttonsHeight = 80.f;

		int _guiId = 0;
		std::shared_ptr<bool> _sharedWndPtr;
		TextChar _textBuffer;
		Quest::Ptr _selectQuest;

		//...
		std::vector<EditorCommand> _editorCommands;
		std::vector<const char*> _listCommands;
		//...
		std::unordered_map<std::string, std::vector<const char*>> _mapLists;
		//...
		EditorList observerLists;
	};
}
