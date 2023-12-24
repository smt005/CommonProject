#pragma once

class MySystem;

class MainUI {
public:
	static void Open(MySystem* mySystem);
	static void Hide();
	
	static void DrawOnSpace();
	static bool IsLockAction();
	static unsigned int GetViewType();

private:
	static void InitCallback();

private:
	static MySystem* _mySystem;
};
