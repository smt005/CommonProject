// ◦ Xyz ◦
#pragma once

#include <string>
#include <vector>
#include "Quest.h"
#include <MyStl/Singleton.h>

class QuestManagerImpl
{
public:
	virtual ~QuestManagerImpl() = default;

	void SetState(const std::string& name, Quest::State state);
	Quest::State GetState(const std::string& name);
	void ActivateState(const std::string& name, Quest::State state);
	Quest::Ptr& Add(const std::string& classQuest, const std::string& nameQuest);
	void Add(const Quest::Ptr& questPtr);
	void Remove(const std::string& name);
	Quest::Ptr GetQuest(const std::string& name);
	const std::vector<std::string>& GetListClasses();
	std::string GetClassName(Quest::Ptr& questPtr);
	bool HasQuest(const std::string& name);
	Quest::State StateFromString(const std::string& stateStr);
	void Load(const std::string& pathFileName);
	void Reload();
	void Save(const std::string& pathFileName = std::string());
	void Clear();
	void Update();

	std::vector<Quest::Ptr>& GetQuests()
	{
		return quests;
	}

	const std::string& PathFileName()
	{
		return _pathFileName;
	}

	void RunCommands(const std::string& questName, const std::string& commandName);
	void SetParamValue(const std::string& questName, const std::string& nameValue, const std::string& valueStr);
	void SetGlobalParamValue(const std::string& nameValue, const std::string& valueStr);

private:
	Quest::Ptr activeQuestPtr;
	std::string _pathFileName;
	std::vector<Quest::Ptr> quests;
};

class QuestManager : public mystd::Singleton<QuestManagerImpl>
{};
