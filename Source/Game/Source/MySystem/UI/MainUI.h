// ◦ Xyz ◦
#pragma once

#include <string>

class MainUI {
public:
	static void Open();
	static void Hide();
	
	static void DrawOnSpace();
	static bool IsLockAction();
	static unsigned int GetViewType();

	static void SetCursorModel(const std::string& nameModel);

private:
	static void InitCallback();
};
