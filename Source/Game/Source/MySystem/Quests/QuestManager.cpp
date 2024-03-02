#include "QuestManager.h"

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

void QuestManager::Load() {
}

void QuestManager::Update() {
}
