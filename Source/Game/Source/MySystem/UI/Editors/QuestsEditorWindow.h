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
			EditorCommand(const std::string& _name) : name(_name) {}
			char* data() {
				return name.data();
			}
			static std::string emptyName;
		};

		template<typename T>
		struct EditorListT {
			std::vector<T> dataList;
			std::vector<const char*> viewList;
			int currentIndex = -1;

			T& Add(T&& text) {
				return dataList.emplace_back(text);
			}
			void Add(std::initializer_list<T>&& texts) {
				dataList.reserve(dataList.size() + texts.size());
				dataList.insert(dataList.end(), texts.begin(), texts.end());
			}
			void MakeViewData() {
				viewList.clear();
				for (T& text : dataList) {
					viewList.emplace_back(text.data());
				}
			}
			void Reserve(int reserveSize) {
				dataList.reserve(reserveSize);
				viewList.clear();
			}
			T& Get() {
				return GetByIndex(currentIndex);
			}
			T& GetByIndex(const int index) {
				if (index < 0 || index >= dataList.size()) {
					static T empty;
					return empty;
				}
				return dataList[index];
			}
			int& GetIndex(const std::string& text)
			{
				currentIndex = -1;
				for (auto&& item : dataList) {
					++currentIndex;

					if (text == item.data()) {
						return currentIndex;
					}
				}
				
				currentIndex = -1;
				return currentIndex;
			}
			void Clear() {
				dataList.clear();
				viewList.clear();
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
		void DrawQuestParams(std::map<std::string, std::string>& paramMap, const std::string& title);
		void ChangeParamDisplay(std::map<std::string, std::string>& paramMap, const std::string& name);
		void QuestButtonDisplay();
		void QuestListButtonDisplay();

		void DrawCommands(Commands& commands, const std::string& title, const std::pair<float, float>& offset);
		void DrawParams(std::vector<std::string>& parameters);
		void ButtonDisplay();
		void PrepareDraw(Quest::Ptr& selectQuest);

		void AddQuest();
		void CopyQuest(Quest::Ptr& questPtr);
		void RemoveQuest(Quest::Ptr& questPtr);
		void AddObserver();
		void Clear();

		void LoadEditorDatas();
		void EditorDatasParceFile(const std::string& filePathHame);
		void PrepareCommand(const std::vector<std::string>& words);

		inline std::shared_ptr<bool> GetSharedWndPtr() {
			if (!_sharedWndPtr) {
				_sharedWndPtr = std::make_shared<bool>(true);
			}
			return _sharedWndPtr;
		}

	private:
		float _width = 1000.f;
		float _height = 700.f;
		float _widthList = 263.f;
		float _widthQuest = 500.f;
		float _topHeight = 100.f;
		float _buttonsWidth = 200.f;
		float _buttonsHeight = 80.f;

		int _guiId = 0;
		std::shared_ptr<bool> _sharedWndPtr;
		TextChar _textBuffer;
		Quest::Ptr _selectQuest;

		EditorListT<EditorCommand> _editorCommands;
		std::unordered_map<std::string, EditorListT<std::string>> _mapLists;

		private:
			static std::string questClassesType;
			static std::string observesType;
	};
}
