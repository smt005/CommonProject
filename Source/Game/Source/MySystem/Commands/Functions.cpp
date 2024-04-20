// ◦ Xyz ◦
#include "Functions.h"
#include "Functions/Actions.h"

// Common
#include <MyStl/Event.h>
#include "MySystem/MySystem.h"
#include "../Objects/Space.h"
#include "../../MySystem/Objects/SpaceManager.h"
#include "../Commands/Events.h"

#include "../../CUDA/Source/Wrapper.h"
#include <iostream>
#include <glm/vec3.hpp>
#include <Object/Color.h>

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
#include "../Commands/Functions/QuestCondition.h"

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

	float StrToFloat(const std::string& valueStr, float min, float max)
	{
		float value = atof(valueStr.c_str());
		value = value < min ? min : value;
		value = value > max ? max : value;
		return value;

	}

	int StrToInt(const std::string& valueStr, int min, int max)
	{
		int value = atoi(valueStr.c_str());
		value = value < min ? min : value;
		value = value > max ? max : value;
		return value;
	}

	/// RunCommandsFromFile string
	void RunCommandsFromFile(const std::string& filePathName)
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

	/// OpenWindow /RewardWindow/CommandsWindow/QuestsWindow/QuestsEditorWindow #QUESTS #QUESTS Text
	void OpenWindow(const std::vector<std::string>& parameters)
	{
		if (parameters.empty()) {
			return;
		}

		const std::string& classWindow = parameters.front();

		if (classWindow == "RewardWindow") {
			if (!UI::ShowingWindow<RewardWindow>()) {
				if (parameters.size() >= 4) {
					UI::ShowWindow<RewardWindow>(parameters[1], parameters[2], parameters[3]);
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

	/// ShowImage #MODELS
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

	/// HideImage #MODELS
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

	/// SetSkyBox #MODELS
	void SetSkyBox(const std::string& modelName)
	{
		if (MySystem::currentSpace) {
			if (ModelPtr& modelPtr = Model::getByName(modelName)) {
				MySystem::currentSpace->SetSkyBoxModel(modelName);
			}
		}
	}

	/// GravityPoints /Enable/Disable
	void GravityPoints(const std::string& param) {
		MySystem::gravitypoints = param == "Enable";
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

	/// AddBodyToPos #MODELS number number number number number number number
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

	/// AddBodyToMousePos #MODELS /ToMousePos/ToCenterSpace number number number number /Default/ContrastRandom/Random/RED/GREEN/BLUE/WHITE/BLACK/Custom
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

		Color color = countParams < 7 ? (1.f, 1.f, 1.f, 1.f) : (0.f, 0.f, 0.f, 1.f);

		if (countParams >= 7) {
			const std::string& typeColor = parameters[6];
			if (typeColor.empty()) {
				//...
			}
			else if (typeColor == "ContrastRandom") {
				int componentColor = help::random_i(0, 2);
				float min = 0.f;
				float max = 0.5f;

				switch (componentColor) {
				{
				case 0: {
					color.setRed(1.f);
					color.setGreen(help::random(min, max));
					color.setBlue(help::random(min, max));
				} break;
				case 1: {
					color.setRed(help::random(min, max));
					color.setGreen(1.f);
					color.setBlue(help::random(min, max));
				} break;
				case 2: {
					color.setRed(help::random(min, max));
					color.setGreen(help::random(min, max));
					color.setBlue(1.f);
				} break;
				default:
					break; }
				}
			}
			else if (typeColor == "Random") {
				if (countParams >= 9) {
					float min = StrToFloat(parameters[7]);
					float max = StrToFloat(parameters[8]);

					color.setRed(help::random(min, max));
					color.setGreen(help::random(min, max));
					color.setBlue(help::random(min, max));
				}
				else {
					color.setRed(help::random(0.f, 0.5f));
					color.setGreen(help::random(0.f, 0.5f));
					color.setBlue(help::random(0.f, 0.5f));
				}
			}
			else if (typeColor == "RED") {
				color.setRed(1.f);
					color.setGreen(0.f);
					color.setBlue(0.f);
			}
			else if (typeColor == "GREEN") {
					color.setRed(0.f);
				color.setGreen(1.f);
					color.setBlue(0.f);
			}
			else if (typeColor == "BLUE") {
					color.setRed(1.f);
					color.setGreen(0.f);
				color.setBlue(1.f);
			}
			else if (typeColor == "WHITE") {
				color.setRed(1.f);
				color.setGreen(1.f);
				color.setBlue(1.f);
			}
			else if (typeColor == "BLACK") {
				color.setRed(0.f);
				color.setGreen(0.f);
				color.setBlue(0.f);
			}
			else if (countParams >= 12 && typeColor == "Custom") {
				color.setRed(StrToFloat(parameters[7]));
				color.setGreen(StrToFloat(parameters[8]));
				color.setBlue(StrToFloat(parameters[9]));
			}
		}

		//...
		SpaceManager::AddObject(nameModel, pos, vel, mass, color);
	}

	/// StartQuest #QUESTS
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

	void LockAction(const std::string& stateStr)
	{
		if (stateStr == "Push") {
			CommonData::PushLockScreen();
		}
		else if (stateStr == "Pop") {
			CommonData::PopLockScreen();
		}
		else if (stateStr == "Unlock") {
			CommonData::UnlockScreen();
		}
	}

	/// SetActiveQuest #QUESTS /ACTIVE/DEACTIVE

	//..................................................................
	void Run(const Command& comand)
	{
		const std::string& comandId = comand.id;
		CommandLog(comand);

		if (comandId == "SetActiveQuest") {
			if (comand.parameters.size() >= 2) {
				QuestManager::ActivateState(comand.parameters[0], QuestManager::StateFromString(comand.parameters[1]));
			}
		}
		else if (comandId == "SetStateQuest") {
			if (comand.parameters.size() >= 2) {
				QuestManager::SetState(comand.parameters[0], QuestManager::StateFromString(comand.parameters[1]));
			}
		}
		else if (comandId == "StartQuest") {
			if (!comand.parameters.empty()) {
				StartQuest(comand.parameters.front());
			}
		}
		else if (comandId == "LoadQuests") {
			if (!comand.parameters.empty()) {
				QuestManager::Load(comand.parameters.front());
			}
		}
		else if (comandId == "ClearQuests") {
			QuestManager::Clear();
		}
		else if (comandId == "SetProcess") {
			if (!comand.parameters.empty()) {
				SetProcess(comand.parameters.front());
			}
		}
		else if (comandId == "SetMultithread") {
			if (!comand.parameters.empty()) {
				SetMultithread(comand.parameters.front());
			}
		}
		else if (comandId == "SetClearColor") {
			SetClearColor(comand.parameters);
		}
		else if (comandId == "OpenWindow") {
			if (!comand.parameters.empty()) {
				OpenWindow(comand.parameters);
			}
		}
		else if (comandId == "CloseWindow") {
			if (!comand.parameters.empty()) {
				CloseWindow(comand.parameters.front());
			}
		}
		else if (comandId == "ShowImage") {
			if (!comand.parameters.empty()) {
				ShowImage(comand.parameters.front());
			}
		}
		else if (comandId == "HideImage") {
			if (!comand.parameters.empty()) {
				HideImage(comand.parameters.front());
			}
		}
		else if (comandId == "ShowText") {
			if (!comand.parameters.empty()) {
				ShowText(comand.parameters.front());
			}
		}
		else if (comandId == "HideText") {
			HideText();
		}
		else if (comandId == "GravityPoints") {
			if (!comand.parameters.empty()) {
				GravityPoints(comand.parameters.front());
			}
		}
		else if (comandId == "ClearSpace") {
			ClearSpace();
		}
		else if (comandId == "ClearAll") {
			ClearAll();
		}
		else if (comandId == "CreateSpace") {
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
		else if (comandId == "RunCommandsFromFile") {
			if (!comand.parameters.empty()) {
				RunCommandsFromFile(comand.parameters.front());
			}
		}
		else if (comandId == "CountOfIteration") {
			if (!comand.parameters.empty()) {
				SetCountOfIteration(comand.parameters.front());
			}
		}
		else if (comandId == "FadeModel") {
			if (comand.parameters.size() >= 2) {
				FadeModel(comand.parameters[0], StrToFloat(comand.parameters[1], 1.f, 10000.0f));
			}
		}
		else if (comandId == "DelayActionCommand") {
			if (comand.parameters.size() >= 3) {
				DelayActionCommand(comand.parameters[0], comand.parameters[1], StrToFloat(comand.parameters[2], 0.f, 1000000.0f));
			}
		}
		else if (comandId == "RunCommands") {
			if (comand.parameters.size() >= 2) {
				QuestManager::RunCommands(comand.parameters[0], comand.parameters[1]);
			}
		}
		else if (comandId == "RunCommandIf") {
			if (comand.parameters.size() >= 7) {
				quest::RunCommandIf(comand.parameters[0],
					                comand.parameters[1],
									comand.parameters[2],
									comand.parameters[3],
									comand.parameters[4],
									comand.parameters[5],
									comand.parameters[6]);
			}
		}
		else if (comandId == "ValueOperation") {
			if (comand.parameters.size() >= 7) {
				quest::ValueOperation(comand.parameters[0],
									  comand.parameters[1],
									  comand.parameters[2],
									  comand.parameters[3],
									  comand.parameters[4],
									  comand.parameters[5],
									  comand.parameters[6]);
			}
		}
		else if (comandId == "LockAction") {
			if (!comand.parameters.empty()) {
				LockAction(comand.parameters.front());
			}
		}
	}
}
