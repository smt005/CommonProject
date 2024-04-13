// ◦ Xyz ◦
#pragma once

#include <string>

namespace quest {
	enum class Expression {
		is_error,
		is_more,
		is_more_or_equal,
		is_less,
		is_less_or_equal,
		is_equal,
		is_not_equal
	};

	bool count_bodies(const std::string& expressions, int number);
	bool max_speed_body(const std::string& expressions, float speed);

	/// RunCommandIf #QUESTS !PARAMS #EXPRESSIONS #QUEST !PARAMS #QUEST !COMMANDS
	void RunCommandIf(const std::string& questNameLeft, const std::string& paramLeft,
					const std::string& expressionStr,
					const std::string& questNameRight, const std::string& paramRight,
					const std::string& questName, const std::string& commandName);
}
