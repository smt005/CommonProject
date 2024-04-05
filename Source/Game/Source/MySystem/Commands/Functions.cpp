// ◦ Xyz ◦
#include "Functions.h"

// Common
#include <MyStl/Event.h>
#include "MySystem/MySystem.h"
#include "../Objects/Space.h"
#include "../../MySystem/Objects/SpaceManager.h"
#include "../Commands/Events.h"

#include "../../CUDA/Source/Wrapper.h"
#include <iostream>
#include <glm/vec3.hpp>

// View
#include "Draw2/Draw2.h"
#include "Draw/Camera/Camera.h"

// Windows
#include "../UI/RewardWindow.h"
#include "../UI/Editors/CommandsWindow.h"
#include "../UI/Editors/QuestsWindow.h"
#include "../UI/CommonData.h"
#include "../UI/Editors/QuestsEditorWindow.h"

// Quest
#include "../Quests/Quest.h"
#include "../Quests/QuestManager.h"

namespace commands
{
	void CommandLog(const Command& command)
	{
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

	/// RunCommands string
	void RunCommands(const std::string& filePathName)
	{
		CommandManager::Run(filePathName);
	}

	/// SetProcess /CPU/GPU
	void SetProcess(const std::string& stateStr)
	{
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

	/// CountOfIteration number
	void SetCountOfIteration(const std::string& countStr)
	{
		if (MySystem::currentSpace) {
			int count = atoi(countStr.c_str());
			if (count >= 0) {
				MySystem::currentSpace->countOfIteration = (size_t)count;
			}
		}
	}

	/// SetMultithread /true/false
	void SetMultithread(const std::string stateStr)
	{
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

	/// SetClearColor number number number number
	void SetClearColor(const std::vector<std::string>& strColors)
	{
		if (strColors.size() < 3) {
			return;
		}

		float r, g, b, a = 0.f;

		size_t countParams = strColors.size();

		r = countParams > 0 ? atof(strColors[0].c_str()) : 0.f;
		g = countParams > 1 ? atof(strColors[1].c_str()) : 0.f;
		b = countParams > 2 ? atof(strColors[2].c_str()) : 0.f;
		a = countParams > 3 ? atof(strColors[3].c_str()) : 0.f;

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

	/// OpenWindow /RewardWindow/CommandsWindow/QuestsWindow/QuestsEditorWindow
	void OpenWindow(const std::vector<std::string>& parameters)
	{
		if (parameters.empty()) {
			return;
		}

		const std::string& classWindow = parameters.front();

		if (classWindow == "RewardWindow") {
			if (!UI::ShowingWindow<RewardWindow>()) {
				if (parameters.size() >= 3) {
					UI::ShowWindow<RewardWindow>(parameters[1], parameters[2]);
				}
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
		else if (classWindow == "QuestsEditorWindow") {
			if (!UI::ShowingWindow<Editor::QuestsEditorWindow>()) {
				UI::ShowWindow<Editor::QuestsEditorWindow>();
			}
		}

		/*else if (classWindow == "XXX") {
			if (!UI::ShowingWindow<XXX>()) {
				ShowWindow<XXX>();
			}
		}*/
	}

	/// CloseWindow /QuestsEditorWindow/RewardWindow/CommandsWindow/QuestsWindow
	void CloseWindow(const std::string& classWindow)
	{
		if (classWindow == "RewardWindow") {
			UI::CloseWindowT<RewardWindow>();
		}
		else if (classWindow == "CommandsWindow") {
			UI::CloseWindowT<CommandsWindow>();
		}
		else if (classWindow == "QuestsWindow") {
			UI::CloseWindowT<QuestsWindow>();
		}
		else if (classWindow == "QuestsEditorWindow") {
			UI::CloseWindowT<Editor::QuestsEditorWindow>();
		}

		/*else if (classWindow == "XXX") {
			CloseWindowT<XXX>();
		}*/
	}

	/// ShowImage #MODEL
	void ShowImage(const std::string& nameModel)
	{
		if (std::find_if(CommonData::nameImageList.begin(), CommonData::nameImageList.end(), [&nameModel](const std::string& itName) {
				return itName == nameModel;
			})
			!= CommonData::nameImageList.end()) {
			return;
		}
		CommonData::nameImageList.emplace_back(nameModel);
	}

	/// HideImage #MODEL
	void HideImage(const std::string& nameModel)
	{
		auto it = std::find_if(CommonData::nameImageList.begin(), CommonData::nameImageList.end(), [&nameModel](const std::string& itName) {
			return itName == nameModel;
		});
		if (it != CommonData::nameImageList.end()) {
			CommonData::nameImageList.erase(it);
		}
	}

	/// ShowText string
	void ShowText(const std::string& text)
	{
		CommonData::textOnScreen = text;
	}

	/// HideText
	void HideText()
	{
		CommonData::textOnScreen.clear();
	}

	/// SetSkyBox #MODEL
	void SetSkyBox(const std::string& modelName)
	{
		if (MySystem::currentSpace) {
			if (ModelPtr& modelPtr = Model::getByName(modelName)) {
				MySystem::currentSpace->SetSkyBoxModel(modelName);
			}
		}
	}

	/// ClearSpace
	void ClearSpace()
	{
		if (MySystem::currentSpace) {
			MySystem::currentSpace->_bodies.clear();
			MySystem::currentSpace->Preparation();
		}
	}

	/// ClearAll
	void ClearAll()
	{
		MySystem::currentSpace.reset();
		CommonData::textOnScreen.clear();
		CommonData::nameImageList.clear();

		for (Quest::Ptr& questPtr : QuestManager::GetQuests()) {
			questPtr->SetState(Quest::State::NONE);
			const std::string& questName = questPtr->Name();

			EventOnTap::Instance().Remove(questName);
			EventOnUpdate::Instance().Remove(questName);
		}
	}

	/// CreateSpace string /Try/Force
	void CreateSpace(const std::vector<std::string>& params)
	{
		const std::string& name = params[0];

		bool createForce = true;
		if (params.size() >= 2) {
			createForce = params[0] == "Force";
		}

		if (createForce) {
			MySystem::currentSpace.reset();
		}

		if (!MySystem::currentSpace) {
			MySystem::currentSpace = SpaceManager::Load(name);
			MySystem::currentSpace->Preparation();
		}
	}

	/// AddBodyToPos #MODEL number number number number number number number
	void AddBodyToPos(const std::vector<std::string>& parameters)
	{
		// model, pos, vel, mass
		size_t countParams = parameters.size();

		const std::string tempStr;
		const std::string& nameModel = countParams >= 1 ? parameters[0] : tempStr;

		Math::Vector3 pos(0.f);
		Math::Vector3 vel(0.f);

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

		float mass = countParams >= 8 ? atof(parameters[7].c_str()) : 1.f;

		//...
		SpaceManager::AddObject(nameModel, pos, vel, mass);
	}

	/// AddBodyToMousePos #MODEL /ToMousePos/ToCenterSpace number number number number
	void AddBodyToMousePos(const std::vector<std::string>& parameters)
	{
		// model, pos(mouse), vel, mass
		size_t countParams = parameters.size();

		const std::string tempStr;
		const std::string& nameModel = countParams >= 1 ? parameters[0] : tempStr;

		Math::Vector3 pos(0.f); // ToCenterSpace
		Math::Vector3 vel(0.f);

		if (countParams >= 2 && parameters[1] == "ToMousePos") {
			auto posTmp = Camera::GetLink().corsorCoord();
			pos.x = posTmp.x;
			pos.y = posTmp.y;
			pos.z = posTmp.z;
		}

		if (countParams >= 5) {
			vel.x = atof(parameters[2].c_str());
			vel.y = atof(parameters[3].c_str());
			vel.z = atof(parameters[4].c_str());
		}

		float mass = countParams >= 6 ? atof(parameters[5].c_str()) : 1.f;

		//...
		SpaceManager::AddObject(nameModel, pos, vel, mass);
	}

	/// StartQuest #QUEST
	void StartQuest(const std::string& name) {
		if (Quest::Ptr questPtr = QuestManager::GetQuest(name)) {
			Commands& commandsDebug = questPtr->_commandsDebug;
			Commands commands;

			commands.reserve(2 + questPtr->_commandsDebug.size()); // +2 для ClearAll, SetActiveQuest
			commands.emplace_back("ClearAll");
			commands.insert(commands.end(), commandsDebug.begin(), commandsDebug.end());
			commands.emplace_back("SetActiveQuest", Parameters{ questPtr->Name(), "ACTIVE" });

			CommandManager::Run(std::forward<Commands>(commands));
		}
	}

	/// SetActiveQuest #QUEST /ACTIVE/DEACTIVE

	//..................................................................
	void Run(const Command& comand)
	{
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
		else if (comandId == "StartQuest") {
			if (!comand.parameters.empty()) {
				CommandLog(comand);
				StartQuest(comand.parameters.front());
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
				OpenWindow(comand.parameters);
			}
		}
		else if (comandId == "CloseWindow") {
			CommandLog(comand);
			if (!comand.parameters.empty()) {
				CloseWindow(comand.parameters.front());
			}
		}
		else if (comandId == "ShowImage") {
			CommandLog(comand);
			if (!comand.parameters.empty()) {
				ShowImage(comand.parameters.front());
			}
		}
		else if (comandId == "HideImage") {
			CommandLog(comand);
			if (!comand.parameters.empty()) {
				HideImage(comand.parameters.front());
			}
		}
		else if (comandId == "ShowText") {
			CommandLog(comand);
			if (!comand.parameters.empty()) {
				ShowText(comand.parameters.front());
			}
		}
		else if (comandId == "HideText") {
			CommandLog(comand);
			HideText();
		}
		else if (comandId == "ClearSpace") {
			ClearSpace();
		}
		else if (comandId == "ClearAll") {
			ClearAll();
		}
		else if (comandId == "CreateSpace") {
			CommandLog(comand);
			if (!comand.parameters.empty()) {
				CreateSpace(comand.parameters);
			}
		}
		else if (comandId == "AddBodyToPos") {
			AddBodyToPos(comand.parameters);
		}
		else if (comandId == "AddBodyToMousePos") {
			AddBodyToMousePos(comand.parameters);
		}
		else if (comandId == "RunCommands") {
			if (!comand.parameters.empty()) {
				RunCommands(comand.parameters.front());
			}
		}
		else if (comandId == "QuestCondition") {
			QuestManager::Condition(comand.parameters);
		}
		else if (comandId == "CountOfIteration") {
			if (!comand.parameters.empty()) {
				SetCountOfIteration(comand.parameters.front());
			}
		}
	}
}
