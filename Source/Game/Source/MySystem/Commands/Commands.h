#pragma once
#include <string>
#include <vector>
#include "json/json.h"

struct Command final {
	Command() = default;
	Command(const std::string& _id)
		: id(_id)
	{}
	Command(const std::string& _id, const std::vector<std::string>& _parameters)
		: id(_id)
		, parameters(_parameters)
	{}
	Command(std::string&& _id, std::vector<std::string>&& _parameters) {
		std::swap(id, _id);
		std::swap(parameters, _parameters);
	}

	bool disable = false;
	std::string tag;

	std::string id;
	std::vector<std::string> parameters;
};

using Commands = std::vector<Command>;

class CommandManager final {
public:
	static Commands Load(const std::string& pathFileName);
	static Commands Load(const Json::Value& valueData);
	static void Run(const std::string& pathFileName);
	static void Run(const Commands& commands);
	static void Run(const Command& command);
};
