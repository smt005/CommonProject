// ◦ Xyz ◦
#pragma once
#include "Commands.h"

namespace commands {
	void Run(const Command& comand);
	const std::vector<const char*>& GetListCommands();
	//const std::vector<const std::string>& GetListCommands();
}
