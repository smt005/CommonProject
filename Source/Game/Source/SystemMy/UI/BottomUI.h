#pragma once

#include "ImGuiManager/UI.h"
#include <functional>

class SystemMy;

using FunAction = std::pair<std::string, std::function<void(void)>>;
using FunActions = std::vector< FunAction>;

enum class AddBodyType {
	ORBIT,
	DIRECT,
	ORBIT_CENTER_MASS,
	NONE
};

class AddObjectUI final : public UI::Window {
public:
	AddObjectUI() : UI::Window(this) { Close(); }
	AddObjectUI(const FunActions& funActionsArg);
	void OnOpen() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 0.f;
	float _y = 60.f;
	float _width = 65.f;
	float _height = 220.f;

	FunActions _funActions;
};

class SetViewUI final : public UI::Window {
public:
	SetViewUI() : UI::Window(this) { Close(); }
	SetViewUI(const FunActions& funActionsArg);
	void OnOpen() override;
	void Update() override;
	void Draw() override;

private:
	float _x = 0.f;
	float _y = 60.f;
	float _width = 65.f;
	float _height = 220.f;

	FunActions _funActions;
};

class BottomUI final : public UI::Window {
	friend AddObjectUI;
	friend SetViewUI;

public:
	BottomUI() : UI::Window(this) { Close(); }
	BottomUI(SystemMy* systemMy);
	void OnOpen() override;
	void OnClose() override;
	void Update() override;
	void Draw() override;

private:
	void GenerateFunAddObjectUI();
	void GenerateFunViewUI();

private:
public:
	float _x = 0.f;
	float _y = 0.f;
	float _width = 0.f;
	float _height = 60.f;

	int timeSpeed = 1;
	SystemMy* _systemMy = nullptr;

	AddBodyType _addBodyType = AddBodyType::NONE;
	bool _lockAddObject = false;
	FunAction _funAddObject;

	bool _lockSetView = false;
	FunAction _funSetView;
	
};
