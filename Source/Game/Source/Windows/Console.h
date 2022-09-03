#pragma once

#include "ImGuiManager/UI.h"

namespace Editor {
	class Console final: public UI::Window {
	public:
		Console();
		~Console();
		void Draw() override;
		inline static bool IsLock() { return lock; }

	private:
		void Demo();
		void Commands();

	private:
		char* _buf;
		int _guiId;

		bool _show_demo_window = true;
		bool _show_another_window = false;
		int _counter = 0;
		bool _showDemo = false;

	public:
		static bool lock;
	};
}