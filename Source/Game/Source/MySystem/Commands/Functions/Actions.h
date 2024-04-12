// ◦ Xyz ◦
#pragma once
#include <string>

namespace commands {
	/// FadeModel #MODELS float
	void FadeModel(const std::string& model, float fadeTime);

	/// DelayActionCommand !COMMANDS quest float
	void DelayActionCommand(const std::string& questName, const std::string& commandsName, float delay);
}
