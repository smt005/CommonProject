// ◦ Xyz ◦
#pragma once
#include <string>

namespace commands {
	/// FadeModel #MODEL float
	void FadeModel(const std::string& model, float fadeTime);

	/// DelayActionCommand !COMMANDS quest float
	void DelayActionCommand(const std::string& questName, const std::string& commandsName, float delay);
}
