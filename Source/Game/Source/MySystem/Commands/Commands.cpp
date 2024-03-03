#include "Commands.h"
#include "Common/Help.h"
#include "Functions.h"

void CommandManager::Run(const std::string& pathFileName) {
	Json::Value valueData;

	if (!help::loadJson(pathFileName, valueData)) {
		return;
	}

	if (!valueData.isArray()) {
		return;
	}

	Commands commands;

	for (auto& jsonCommand : valueData) {
		if (!jsonCommand.isObject()) {
			continue;
		}

		Json::Value& jsonId = jsonCommand["id"];
		const std::string id = jsonId.asString();
		if (id.empty()) {
			continue;
		}

		Command& command = commands.emplace_back(id);

		Json::Value& jsonDisable = jsonCommand["disable"];
		command.disable = (!jsonDisable.empty() && jsonDisable.isBool()) ? jsonDisable.asBool() : false;

		Json::Value& jsonParams = jsonCommand["params"];
		if (jsonParams.empty() || !jsonParams.isArray()) {
			continue;
		}

		for (auto& jsonParam : jsonParams) {
			if (jsonParam.isString()) {
				command.parameters.emplace_back(jsonParam.asString());
			}
		}
	}

	CommandManager::Run(commands);
}

void CommandManager::Run(const Commands& commands) {
	for (const Command& command : commands) {
		Run(command);
	}
}

void CommandManager::Run(const Command& command) {
	if (!command.disable) {
		commands::Run(command);
	}
}
