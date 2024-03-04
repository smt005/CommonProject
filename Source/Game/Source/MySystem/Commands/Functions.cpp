#include "Functions.h"

// Common
#include "MySystem/MySystem.h"
#include "../Objects/Space.h"
#include "../Quests/QuestManager.h"
#include "../../CUDA/Source/Wrapper.h"
#include <iostream>

// View
#include "Draw2/Draw2.h"

// Windows
#include "../UI/RewardWindow.h"

namespace commands {
void CommandLog(const Command& command) {
	if (command.tag.empty()) {
		std::cout << command.id << ": [";
	}
	else {
		std::cout << command.tag << ":" << command.id << ": [";
	}

	size_t size = command.parameters.size() - 1;
	for (const std::string& param : command.parameters) {
		std::cout << param;
		if (size > 0) {
			std::cout << ", ";
			--size;
		}
	}

	std::cout << ']' << std::endl;
}

void SetProcess(const std::string stateStr) {
	// CPU, GPU
	if (stateStr == "CPU") {
		CUDA::processGPU = false;
	}
	else if (stateStr == "GPU") {
		CUDA::processGPU = true;
	}

	if (MySystem::currentSpace) {
		MySystem::currentSpace->processGPU = CUDA::processGPU;
	}
}

void SetMultithread(const std::string stateStr) {
	// true, false
	if (stateStr == "true") {
		CUDA::multithread = true;
	}
	else if (stateStr == "false") {
		CUDA::multithread = false;
	}

	if (MySystem::currentSpace) {
		MySystem::currentSpace->multithread = CUDA::multithread;
	}
}

// View
void SetClearColor(const std::vector<std::string>& strColors) {
	if (strColors.size() < 3) {
		return;
	}

	float r, g, b, a = 0.f;

	r = atof(strColors[0].c_str());
	g = atof(strColors[1].c_str());
	b = atof(strColors[2].c_str());
	a = atof(strColors[3].c_str());

	r = r > 0.f ? r : 0.f;
	g = g > 0.f ? g : 0.f;
	b = b > 0.f ? b : 0.f;
	a = a > 0.f ? a : 0.f;

	r = r < 1.f ? r : 1.f;
	g = g < 1.f ? g : 1.f;
	b = b < 1.f ? b : 1.f;
	a = a < 1.f ? a : 1.f;

	Draw2::SetClearColor(r, g, b, a);
}

// Windows
void OpenWindow(const std::string& classWindow) {
	if (classWindow == "RewardWindow") {
		if (!UI::ShowingWindow<RewardWindow>()) {
			UI::ShowWindow<RewardWindow>();
		}
	}
	/*else if (classWindow == "XXX") {
		if (!UI::ShowingWindow<XXX>()) {
			UI::ShowWindow<XXX>();
		}
	}*/
}

//..................................................................
void Run(const Command& comand) {
	const std::string& comandId = comand.id;

	if (comandId == "SetActiveQuest") {
		if (comand.parameters.size() >= 2) {
			CommandLog(comand);
			QuestManager::ActivateState(comand.parameters[0], QuestManager::StateFromString(comand.parameters[1]));
		}
	}
	else if (comandId == "SetStateQuest") {
		if (comand.parameters.size() >= 2) {
			CommandLog(comand);
			QuestManager::SetState(comand.parameters[0], QuestManager::StateFromString(comand.parameters[1]));
		}
	}
	else if (comandId == "LoadQuests") {
		if (comand.parameters.size() >= 1) {
			CommandLog(comand);
			QuestManager::Load(comand.parameters.front());
		}
	}
	else if (comandId == "SetProcess") {
		if (comand.parameters.size() >= 1) {
			CommandLog(comand);
			SetProcess(comand.parameters.front());
		}
	}
	else if (comandId == "SetMultithread") {
		if (comand.parameters.size() >= 1) {
			CommandLog(comand);
			SetMultithread(comand.parameters.front());
		}
	}
	else if (comandId == "SetClearColor") {
		CommandLog(comand);
		SetClearColor(comand.parameters);
	}
	else if (comandId == "OpenWindow") {
		CommandLog(comand);
		if (comand.parameters.size() >= 1) {
			OpenWindow(comand.parameters.front());
		}
	}
}

}
