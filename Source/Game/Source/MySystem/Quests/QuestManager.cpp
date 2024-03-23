// ◦ Xyz ◦
// ◦ Xyz ◦
#include "QuestManager.h"
#include "Quests.h"
#include "Common/Help.h"

std::vector<Quest::Ptr> QuestManager::quests;
Quest::Ptr QuestManager::activeQuestPtr;

void QuestManager::SetState(const std::string& name, Quest::State state) {
	auto it = std::find_if(quests.begin(), quests.end(), [&name](const Quest::Ptr& questPtr) { return questPtr->Name() == name; });
	if (it != quests.end()) {
		activeQuestPtr = *it;
		activeQuestPtr->SetState(state);
	}
}

Quest::State QuestManager::GetState(const std::string& name) {
	auto it = std::find_if(quests.begin(), quests.end(), [&name](const Quest::Ptr& questPtr) { return questPtr->Name() == name; });
	if (it != quests.end()) {
		return (*it)->GetState();
	}
	return Quest::State::NONE;
}

void QuestManager::ActivateState(const std::string& name, Quest::State state) {
	if (Quest::Ptr questPtr = QuestManager::GetQuest(name)) {
		questPtr->ActivateState(state);
	}
}

void QuestManager::Add(const Quest::Ptr& questPtr) {
	quests.emplace_back(questPtr);
}

void QuestManager::Remove(const std::string& name) {
	auto it = std::find_if(quests.begin(), quests.end(), [&name](const Quest::Ptr& questPtr) { return questPtr->Name() == name; });
	if (it != quests.end()) {
		quests.erase(it);
	}
}

Quest::Ptr QuestManager::GetQuest(const std::string& name) {
	auto it = std::find_if(quests.begin(), quests.end(), [&name](const Quest::Ptr& questPtr) { return questPtr->Name() == name; });
	if (it != quests.end()) {
		return *it;
	}

	return Quest::Ptr();
}

bool QuestManager::HasQuest(const std::string& name) {
	return quests.end() == std::find_if(quests.begin(), quests.end(), [&name](const Quest::Ptr& questPtr) { return questPtr->Name() == name; });
}

Quest::State QuestManager::StateFromString(const std::string& stateStr) {
	if (stateStr == "INACTIVE") {
		return Quest::State::INACTIVE;
	}
	else if (stateStr == "ACTIVE") {
		return Quest::State::ACTIVE;
	}
	else if (stateStr == "COMPLETE") {
		return Quest::State::COMPLETE;
	}

	return Quest::State::NONE;
}

void QuestManager::Load(const std::string& pathFileName) {
	Json::Value valueDatas;

	if (!help::loadJson(pathFileName, valueDatas)) {
		return;
	}

	if (!valueDatas.isArray()) {
		return;
	}

	for (auto& valueData : valueDatas) {
		if (!valueData.isObject()) {
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

		Quest* quest = nullptr;

		if (classStr == "QuestStart") {
			quest = quests.emplace_back(new QuestStart(questId)).get();

		} else if (classStr == "QuestSphere100") {
			quest = quests.emplace_back(new QuestSphere100(questId)).get();
		}
		else if (classStr == "QuestSphere") {
			quest = quests.emplace_back(new QuestSphere(questId)).get();
		}
		else {
			quest = quests.emplace_back(new Quest(questId)).get();
		}

		if (quest) {
			quest->SetState(state);

			Json::Value& jsonCommands = valueData["commands"];
			if (!jsonCommands.empty()) {
				quest->_commands = CommandManager::Load(jsonCommands);

				for (Command& conmmand : quest->_commands) {
					conmmand.tag = questId;
				}
			}

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
		}
	}
}

void QuestManager::Clear() {
	quests.clear();
}

void QuestManager::Update() {
}
