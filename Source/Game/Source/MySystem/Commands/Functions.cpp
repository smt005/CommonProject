#include "Functions.h"

// Common
#include "MySystem/MySystem.h"
#include "../Objects/Space.h"
#include "../Quests/QuestManager.h"
#include "../../CUDA/Source/Wrapper.h"
#include <iostream>
#include <glm/vec3.hpp>

// View
#include "Draw2/Draw2.h"
#include "Draw/Camera/Camera.h"

// Windows
#include "../UI/RewardWindow.h"
#include "../UI/Debug/CommandsWindow.h"
#include "../UI/Debug/QuestsWindow.h"

// Space
#include "../../MySystem/Objects/SpaceManager.h"


namespace commands {
void CommandLog(const Command& command) {
	std::cout << "[COMMAND] ";

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

void RunCommands(const std::string& filePathName) {
	CommandManager::Run(filePathName);
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
	else if (classWindow == "CommandsWindow") {
		if (!UI::ShowingWindow<CommandsWindow>()) {
			UI::ShowWindow<CommandsWindow>();
		}
	}
	else if (classWindow == "QuestsWindow") {
		if (!UI::ShowingWindow<QuestsWindow>()) {
			UI::ShowWindow<QuestsWindow>();
		}
	}
	/*else if (classWindow == "XXX") {
		if (!UI::ShowingWindow<XXX>()) {
			UI::ShowWindow<XXX>();
		}
	}*/
}

// Space
void SetSkyBox(const std::string& modelName) {
	if (MySystem::currentSpace) {
		if (ModelPtr& modelPtr = Model::getByName(modelName)) {
			MySystem::currentSpace->SetSkyBoxModel(modelName);
		}
	}
}


void ClearSpace() {
	if (MySystem::currentSpace) {
		MySystem::currentSpace->_bodies.clear();
		MySystem::currentSpace->Preparation();
	}
}

void AddBody(const std::vector<std::string>& parameters) {
	// model, pos, vel, mass
	size_t countParams = parameters.size();

	const std::string tempStr;
	const std::string& nameModel = countParams >= 1 ? parameters[0] : tempStr;

	Math::Vector3 pos(0.f);
	Math::Vector3 vel(0.f);

	if (countParams == 2 && parameters[1] == "mouse") {
		auto posTmp = Camera::GetLink().corsorCoord();
		pos.x = posTmp.x;
		pos.y = posTmp.y;
		pos.z = posTmp.z;

		// vel == 0
	} else {
		if (countParams >= 4) {
			pos.x = atof(parameters[1].c_str());
			pos.y = atof(parameters[2].c_str());
			pos.z = atof(parameters[3].c_str());
		}

		if (countParams >= 7) {
			vel.x = atof(parameters[4].c_str());
			vel.y = atof(parameters[5].c_str());
			vel.z = atof(parameters[6].c_str());
		}
	}

	float mass = countParams >= 8 ? atof(parameters[7].c_str()) : 1.f;

	//...
	SpaceManager::AddObject(nameModel, pos, vel, mass);
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
		if (!comand.parameters.empty()) {
			CommandLog(comand);
			QuestManager::Load(comand.parameters.front());
		}
	}
	else if (comandId == "ClearQuests") {
		CommandLog(comand);
		QuestManager::Clear();
	}
	else if (comandId == "SetProcess") {
		if (!comand.parameters.empty()) {
			CommandLog(comand);
			SetProcess(comand.parameters.front());
		}
	}
	else if (comandId == "SetMultithread") {
		if (!comand.parameters.empty()) {
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
		if (!comand.parameters.empty()) {
			OpenWindow(comand.parameters.front());
		}
	}
	else if (comandId == "ClearSpace") {
		ClearSpace();
	}
	else if (comandId == "AddBody") {
		AddBody(comand.parameters);
	}
	else if (comandId == "RunCommands") {
		if (!comand.parameters.empty()) {
			RunCommands(comand.parameters.front());
		}
	}
}

}
