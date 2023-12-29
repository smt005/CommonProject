#include "Quest.h"
#include <Common/Help.h>

std::map<std::string, Quest::Ptr> Quest::quests;

Quest::Quest(const std::string& name)
	: _name(name) 
{}

// STATIC
bool Quest::Load() {
	std::string filePath = "Quests/Quests.json";
	Json::Value valueData;

	if (!help::loadJson(filePath, valueData) || !valueData.isArray() || valueData.empty()) {
		return false;
	}

	for (Json::Value& jsonQuest : valueData) {
		if (!jsonQuest["name"].isString()) {
			continue;
		}

		std::string name = jsonQuest["name"].asString();
		std::string test;

		if (jsonQuest["test"].isString()) {
			test = jsonQuest["test"].asString();
		}

		auto it = quests.emplace(name, new Quest(name));
		it.first->second->_test = test;
	}

	return true;
}
