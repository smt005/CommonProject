#pragma once

class SystemMy;

class MainUI {
public:
	static void Open(SystemMy* systemMyArg);
	static void Hide();
	static bool IsLockAction();
	static void DrawOnSpace();

private:
	static void InitCallback();

private:
	static SystemMy* systemMy;
};
