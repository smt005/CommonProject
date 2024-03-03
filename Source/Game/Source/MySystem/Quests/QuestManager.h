#pragma once

#include <string>
#include <vector>
#include "Quest.h"

class QuestManager {
public:
	static void SetState(const std::string& name, Quest::State state);
	static Quest::State GetState(const std::string& name);
	static void ActivateState(const std::string& name, Quest::State state);
	static void Add(const Quest::Ptr& questPtr);
	static void Remove(const std::string& name);
	static Quest::Ptr GetQuest(const std::string& name);
	static bool HasQuest(const std::string& name);
	static Quest::State StateFromString(const std::string& stateStr);
	static void Load(const std::string& pathFileName);
	static void Update();

private:
	static Quest::Ptr activeQuestPtr;
	static std::vector<Quest::Ptr> quests;
};
