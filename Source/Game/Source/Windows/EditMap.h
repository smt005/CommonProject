#pragma once

#include "ImGuiManager/UI.h"

namespace Editor {
	class Map final: public UI::Window {
	public:
		Map();
		void Draw() override;
		void OnOpen() override;
		void OnClose() override;

	private:
		bool show_demo_window = true;
		bool show_another_window = false;
		int counter = 0;
		int _guiId;
	};
}