// ◦ Xyz ◦
#pragma once

#include <ImGuiManager/UI.h>
#include <../../../Source/ImGuiManager/Editor/Common/CommonPopupModal.h>
#include <../../../Source/ImGuiManager/Editor/Common/CommonUI.h>
#include <Object/Object.h>

namespace Editor
{
	class MapEditor final : public UI::Window, public CommonPopupModal::Wptr
	{
		friend UI;

	public:
		static const std::string windowName;
		static Object::Ptr NewObject();
		static void AddObjectToMap();

	public:
		MapEditor();
		void Draw() override;
		void OnOpen() override;
		void OnClose() override;
		void Update() override;

	private:
		void ListDisplay();
		void EditDisplay();
		void ButtonDisplay();

		void Load();
		void Save();

	private:
		int _guiId;
		Object::Ptr _selectObjectPtr;
		Object::Ptr _tempObjectPtr;

		Editor::TextChar _name;
		Editor::TextChar _model;
	};
}
