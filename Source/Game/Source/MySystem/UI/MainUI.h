// ◦ Xyz ◦
#pragma once

class MainUI {
public:
	static void Open();
	static void Hide();
	
	static void DrawOnSpace();
	static bool IsLockAction();
	static unsigned int GetViewType();

private:
	static void InitCallback();
};
