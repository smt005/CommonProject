#pragma once
#include <string>
#include <vector>

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
	std::string id;
	std::vector<std::string> parameters;
};

using Commands = std::vector<Command>;

class CommandManager final {
public:
	static void Run(const std::string& pathFileName);
	static void Run(const Commands& commands);
	static void Run(const Command& command);
};
