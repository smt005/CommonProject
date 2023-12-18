#pragma once

class SystemMy;

class MainUI {
public:
	static void Open(SystemMy* systemMyArg);
	static void Hide();
	
	static void DrawOnSpace();
	static bool IsLockAction();
	static unsigned int GetViewType();

private:
	static void InitCallback();

private:
	static SystemMy* systemMy;
};
