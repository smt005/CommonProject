// ◦ Xyz ◦
#pragma once

#include <string>
#include <map>

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

	enum class Operation {
		is_error,
		is_addition,
		is_subtraction,
		is_division,
		is_multiplication/*,
		is_remainder*/
	};

	/// RunCommandIf #QUESTS !PARAMS #EXPRESSIONS #QUESTS !PARAMS #QUESTS !COMMANDS
	void RunCommandIf(const std::string& questNameLeft, const std::string& paramLeft,
					const std::string& expressionStr,
					const std::string& questNameRight, const std::string& paramRight,
					const std::string& questName, const std::string& commandName);

	/// ValueOperation #QUESTS !PARAMS #OPERATIONS #QUESTS !PARAMS #QUESTS !PARAMS
	void ValueOperation(const std::string& questNameLeft, const std::string& paramLeft,
					const std::string& operationStr,
					const std::string& questNameRight, const std::string& paramRight,
					const std::string& questNameResult, const std::string& paramResult);

	// 
	const std::map<std::string, std::string>& GetMapGameParams();
}
