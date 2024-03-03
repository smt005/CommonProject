#include "Functions.h"
#include "../Quests/QuestManager.h"

//..................................................................
namespace commands {
	void Run(const Command& comand) {
		const std::string& comandId = comand.id;

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
		else if (comandId == "LoadQuests") {
			if (comand.parameters.size() >= 1) {
				QuestManager::Load(comand.parameters[0]);
			}
		}
	}
}
