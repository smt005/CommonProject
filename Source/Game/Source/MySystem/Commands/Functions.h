// ◦ Xyz ◦
#pragma once
#include "Commands.h"

namespace commands {
	float StrToFloat(const std::string& valueStr, float min = 0.f, float max = 1.f);
	int StrToInt(const std::string& valueStr, int min = std::numeric_limits<int>::min(), int max = std::numeric_limits<int>::max());

	void RunCommandsFromFile(const std::string& filePathName);
	void SetProcess(const std::string& stateStr);
	void SetCountOfIteration(const std::string& countStr);
	void SetMultithread(const std::string stateStr);
	void SetClearColor(const std::vector<std::string>& strColors);
	void OpenWindow(const std::vector<std::string>& parameters);
	void CloseWindow(const std::string& classWindow);
	void ShowImage(const std::string& nameModel);
	void HideImage(const std::string& nameModel);
	void ShowText(const std::string& text);
	void HideText();
	void SetSkyBox(const std::string& modelName);
	void GravityPoints(const std::string& param);
	void ClearSpace();
	void ClearAll();
	void CreateSpace(const std::vector<std::string>& params);
	void AddBodyToPos(const std::vector<std::string>& parameters);
	void AddBodyToMousePos(const std::vector<std::string>& parameters);
	void StartQuest(const std::string& name);

	void Run(const Command& comand);
}
