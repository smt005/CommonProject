// ◦ Xyz ◦
#pragma once

#include <string>
#include <vector>
#include "Quest.h"

class QuestManager
{
public:
	static void SetState(const std::string& name, Quest::State state);
	static Quest::State GetState(const std::string& name);
	static void ActivateState(const std::string& name, Quest::State state);
	static Quest::Ptr& Add(const std::string& classQuest, const std::string& nameQuest);
	static void Add(const Quest::Ptr& questPtr);
	static void Remove(const std::string& name);
	static Quest::Ptr GetQuest(const std::string& name);
	static const std::vector<std::string>& GetListClasses();
	static std::string GetClassName(Quest::Ptr& questPtr);
	static bool HasQuest(const std::string& name);
	static Quest::State StateFromString(const std::string& stateStr);
	static void Load(const std::string& pathFileName);
	static void Reload();
	static void Save(const std::string& pathFileName = std::string());
	static void Clear();
	static void Update();

	static std::vector<Quest::Ptr>& GetQuests() {
		return quests;
	}

	static void Condition(const std::vector<std::string>& params);
	static void RunCommands(const std::string& questName, const std::string& commandName);

private:
	static Quest::Ptr activeQuestPtr;
	static std::string lastPathFileName; // TODO:
	static std::vector<Quest::Ptr> quests;
};
