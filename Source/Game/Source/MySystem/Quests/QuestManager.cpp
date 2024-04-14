// ◦ Xyz ◦
#include "QuestManager.h"
#include "Quests.h"
#include "QuestCondition.h"
#include "Common/Help.h"

std::vector<Quest::Ptr> QuestManager::quests;
std::string QuestManager::lastPathFileName;
Quest::Ptr QuestManager::activeQuestPtr;

void QuestManager::SetState(const std::string& name, Quest::State state)
{
	auto it = std::find_if(quests.begin(), quests.end(), [&name](const Quest::Ptr& questPtr) { return questPtr->Name() == name; });
	if (it != quests.end()) {
		activeQuestPtr = *it;
		activeQuestPtr->SetState(state);
	}
}

Quest::State QuestManager::GetState(const std::string& name)
{
	auto it = std::find_if(quests.begin(), quests.end(), [&name](const Quest::Ptr& questPtr) { return questPtr->Name() == name; });
	if (it != quests.end()) {
		return (*it)->GetState();
	}
	return Quest::State::NONE;
}

void QuestManager::ActivateState(const std::string& name, Quest::State state)
{
	if (Quest::Ptr questPtr = QuestManager::GetQuest(name)) {
		questPtr->ActivateState(state);
	}
}

Quest::Ptr& QuestManager::Add(const std::string& classQuest, const std::string& nameQuest)
{
	if (classQuest == "QuestStart") {
		return quests.emplace_back(new QuestStart(nameQuest));
	}
	else if (classQuest == "QuestSphere100") {
		return quests.emplace_back(new QuestSphere100(nameQuest));
	}
	else if (classQuest == "QuestSphere") {
		return quests.emplace_back(new QuestSphere(nameQuest));
	}
	
	return quests.emplace_back(new Quest(nameQuest));
}

void QuestManager::Add(const Quest::Ptr& questPtr)
{
	quests.emplace_back(questPtr);
}

void QuestManager::Remove(const std::string& name)
{
	auto it = std::find_if(quests.begin(), quests.end(), [&name](const Quest::Ptr& questPtr) { return questPtr->Name() == name; });
	if (it != quests.end()) {
		quests.erase(it);
	}
}

Quest::Ptr QuestManager::GetQuest(const std::string& name)
{
	auto it = std::find_if(quests.begin(), quests.end(), [&name](const Quest::Ptr& questPtr) { return questPtr->Name() == name; });
	if (it != quests.end()) {
		return *it;
	}

	return Quest::Ptr(new Quest("EMPTY"));
}

const std::vector<std::string>& QuestManager::GetListClasses()
{
	static const std::vector<std::string> listClasses = { "Quest", "QuestStart", "QuestSphere", "QuestSphere100" };
	return listClasses;
}

std::string QuestManager::GetClassName(Quest::Ptr& questPtr)
{
	if (dynamic_cast<QuestStart*>(questPtr.get())) {
		return "QuestStart";
	}
	else if (dynamic_cast<QuestSphere100*>(questPtr.get())) {
		return "QuestSphere100";
	}
	else if (dynamic_cast<QuestSphere*>(questPtr.get())) {
		return "QuestSphere";
	}
	
	return "Quest";
}

bool QuestManager::HasQuest(const std::string& name)
{
	return quests.end() == std::find_if(quests.begin(), quests.end(), [&name](const Quest::Ptr& questPtr) { return questPtr->Name() == name; });
}

Quest::State QuestManager::StateFromString(const std::string& stateStr)
{
	if (stateStr == "DEACTIVE") {
		return Quest::State::DEACTIVE;
	}
	else if (stateStr == "ACTIVE") {
		return Quest::State::ACTIVE;
	}
	else if (stateStr == "COMPLETE") {
		return Quest::State::COMPLETE;
	}

	return Quest::State::NONE;
}

void QuestManager::Load(const std::string& pathFileName)
{
	Json::Value valueDatas;

	if (!help::loadJson(pathFileName, valueDatas)) {
		return;
	}

	if (!valueDatas.isArray()) {
		return;
	}

	lastPathFileName = pathFileName;

	for (auto& valueData : valueDatas) {
		if (!valueData.isObject()) {
			continue;
		}

		Json::Value& jsonGlobalParams = valueData["global_params"];
		if (!jsonGlobalParams.empty() && jsonGlobalParams.isObject()) {
			for (auto&& jsonKey : jsonGlobalParams.getMemberNames()) {
				Json::Value& jsonParam = jsonGlobalParams[jsonKey];

				if (jsonParam.isString()) {
					std::string paramValue = jsonParam.asString();
					Quest::globalParams.emplace(jsonKey, paramValue);
				}			
			}
			continue;
		}

		Json::Value& jsonId = valueData["id"];
		const std::string questId = !jsonId.empty() ? jsonId.asString() : std::string();
		if (questId.empty()) {
			continue;
		}

		Json::Value& jsonClass = valueData["class"];
		const std::string classStr = !jsonClass.empty() ? jsonClass.asString() : "Quest";

		Json::Value& jsonState = valueData["state"];
		const std::string stateStr = !jsonState.empty() ? jsonState.asString() : "NONE";
		Quest::State state = StateFromString(stateStr);

		Quest::Ptr& questPtr = Add(classStr, questId);
		Quest* quest = questPtr.get();

		if (quest) {
			quest->SetState(state);

			auto parceCommande = [&questId](Commands& commands, const std::string& name, const Json::Value& valueData) {
				const Json::Value& jsonCommands = valueData[name];
				if (!jsonCommands.empty()) {
					commands = CommandManager::Load(jsonCommands);

					for (Command& conmmand : commands) {
						conmmand.tag = questId;
					}
				}
			};

			// Commands
			{
				parceCommande(quest->_commandsOnInit, "commands_on_init", valueData);
				parceCommande(quest->_commandsOnTap, "commands_on_tap", valueData);
				parceCommande(quest->_commandsOnUpdate, "commands_on_update", valueData);
				parceCommande(quest->_commandsOnCondition, "commands_on_condition", valueData);
				parceCommande(quest->_commandsDebug, "commands_debug", valueData);

				Json::Value& jsonParams = valueData["params"];
				if (!jsonParams.empty() && jsonParams.isObject()) {
					for (auto&& jsonKey : jsonParams.getMemberNames()) {
						Json::Value& jsonParam = jsonParams[jsonKey];

						if (jsonParam.isString()) {
							std::string param = jsonParam.asString();
							quest->_params.emplace(jsonKey, param);
						}
					}
				}

				Json::Value& jsonDescription = valueData["description"];
				if (jsonDescription.isString()) {
					questPtr->_description = jsonDescription.asString();
				}
			}

			// Commands
			{
				Json::Value& jsonCommonCommands = valueData["commands"];
				if (!jsonCommonCommands.empty() && jsonCommonCommands.isObject()) {
					for (auto&& jsonKey : jsonCommonCommands.getMemberNames()) {
						Json::Value& jsonCommands = jsonCommonCommands[jsonKey];

						Commands& commands = quest->_commandMap[jsonKey];
						parceCommande(commands, jsonKey, jsonCommonCommands);
					}
				}
			}
		}
	}
}

void QuestManager::Reload()
{
	if (!lastPathFileName.empty()) {
		quests.clear();
		Load(lastPathFileName);
	}
}

void QuestManager::Save(const std::string& pathFileName)
{
	const std::string* pathFileNamePtr = nullptr;

	if (!pathFileName.empty()) {
		pathFileNamePtr = &pathFileName;
	}
	else {
		pathFileNamePtr = &lastPathFileName;
	}

	Json::Value valueDatas; // Array

	for (Quest::Ptr& questPtr : quests) {
		Json::Value questJson;

		questJson["id"] = questPtr->Name();
		questJson["class"] = GetClassName(questPtr);

		// Params
		if (!questPtr->_params.empty()) {
			Json::Value paramsJson;
			for (std::pair<const std::string, std::string>& pairParam : questPtr->_params) {
				paramsJson[pairParam.first] = pairParam.second;
			}
			questJson["params"] = paramsJson;
		}

		if (!questPtr->_description.empty()) {
			questJson["description"] = questPtr->_description;
		}

		// Commands
		auto appendCommande = [](Commands& commands, const std::string& name, Json::Value& commandsQuestJson) {
			if (commands.empty()) {
				return;
			}

			Json::Value commandsJson;
		
			for (Command& command : commands) {
				Json::Value commandJson;

				commandJson["id"] = command.id;

				if (command.disable) {
					commandJson["disable"] = command.disable;
				}

				if (!command.parameters.empty()) {
					Json::Value paramsJson;

					for (const std::string& param : command.parameters) {
						paramsJson.append(param);
					}

					commandJson["params"] = paramsJson;
				}

				commandsJson.append(commandJson);
			}

			commandsQuestJson[name] = commandsJson;
		};

		appendCommande(questPtr->_commandsOnInit, "commands_on_init", questJson);
		appendCommande(questPtr->_commandsOnTap, "commands_on_tap", questJson);
		appendCommande(questPtr->_commandsOnUpdate, "commands_on_update", questJson);
		appendCommande(questPtr->_commandsOnCondition, "commands_on_condition", questJson);
		appendCommande(questPtr->_commandsDebug, "commands_debug", questJson);

		Json::Value& commonCommandsJson = questJson["commands"];

		for (auto& pairCommands : questPtr->_commandMap) {
			appendCommande(pairCommands.second, pairCommands.first, commonCommandsJson);
		}

		// Append quest
		valueDatas.append(questJson);
	}

	// global_params
	Json::Value& jsonGlobalParams = valueDatas["global_params"];
	for (std::pair<const std::string, std::string>& paramPair : Quest::globalParams) {
		jsonGlobalParams[paramPair.first] = paramPair.second;
	}

	// Сохранение
	help::saveJson(*pathFileNamePtr, valueDatas, " ");
}

void QuestManager::Clear()
{
	quests.clear();
}

void QuestManager::Update()
{
}

/// QuestCondition #QUESTS /count_boties/max_speed_body/temp_number #EXPRESSIONS number
void QuestManager::Condition(const std::vector<std::string>& params)
{
	if (params.size() < 4) {
		return;
	}

	Quest::Ptr questPtr = QuestManager::GetQuest(params[0]);
	if (!questPtr) {
		return;
	}

	// TODO:
	if (params[1] == "count_bodies") {
		int number = atoi(params[3].c_str());

		if (quest::count_bodies(params[2], number)) {
			CommandManager::Run(questPtr->_commandsOnCondition);
		}
	}
	else if (params[1] == "max_speed_body") {
		float speed = atof(params[3].c_str());

		if (quest::max_speed_body(params[2], speed)) {
			CommandManager::Run(questPtr->_commandsOnCondition);
		}
	}
	else {
		auto itNumber = questPtr->_params.find("temp_value");
		if (itNumber == questPtr->_params.end() && itNumber->second.empty()) {
			return;
		}

		int number = atoi(itNumber->second.c_str());

		if (quest::count_bodies(params[2], number)) {
			CommandManager::Run(questPtr->_commandsOnCondition);
		}
	}
}

/// RunCommands !COMMANDS
void QuestManager::RunCommands(const std::string& questName, const std::string& commandName)
{
	if (Quest::Ptr questPtr = QuestManager::GetQuest(questName)) {
		auto it = questPtr->_commandMap.find(commandName);

		if (it != questPtr->_commandMap.end()) {
			CommandManager::Run(it->second);
		}
	}
}

/// SetParamValue #QUESTS name value
void QuestManager::SetParamValue(const std::string& questName, const std::string& nameValue, const std::string& valueStr)
{
	if (Quest::Ptr questPtr = QuestManager::GetQuest(questName)) {
		questPtr->_params.emplace(nameValue, valueStr);
	}
}

/// SetGlobalParamValue name value
void QuestManager::SetGlobalParamValue(const std::string& nameValue, const std::string& valueStr)
{
	Quest::globalParams.emplace(nameValue, valueStr);
}
