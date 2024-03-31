// ◦ Xyz ◦
#pragma once

#include "ImGuiManager/UI.h"
#include <vector>
#include "Common/Help.h"
#include "../../../Source/ImGuiManager/Editor/Common/CommonPopupModal.h"
#include "../../../Source/ImGuiManager/Editor/Common/CommonUI.h"

namespace Editor {
	class ModelEditor final : public UI::Window, public CommonPopupModal::Wptr {
		friend UI;

	public:
		ModelEditor();
		void Draw() override;
		void OnOpen() override;
		void OnClose() override;

	private:
		void ListDisplay();
		void EditDisplay();
		void ButtonDisplay();

		void Select(std::string& name);
		void OpenExplorer(const std::string& name);
		void Remove(const std::string& name);
		void Add(const std::string& newName);
		void Copy(std::string newName, const std::string copyName);
		void Apply();
		void ChooseFile(Editor::TextChar& textChar);

		void Load();
		void Save();

	private:
		int _guiId;
		std::string* _selectNamePtr = nullptr;
		std::vector<std::string> _modelNames;

		Editor::TextChar _name;
		Editor::TextChar _shape;
		Editor::TextChar _texture;
		Editor::TextChar _physicShape;
		Editor::TextChar _scale;
	};
}
