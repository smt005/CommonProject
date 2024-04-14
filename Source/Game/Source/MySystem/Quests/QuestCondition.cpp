// ◦ Xyz ◦
#include "QuestCondition.h"
#include <MySystem/MySystem.h>
#include "../Commands/Commands.h"
#include "../Objects/Space.h"
#include "Common/Help.h"
#include "Quest.h"
#include "QuestManager.h"

namespace quest
{
	float StrToFloat(const std::string& valueStr, float min, float max)
	{
		float value = atof(valueStr.c_str());
		value = value < min ? min : value;
		value = value > max ? max : value;
		return value;

	}

	double StrToDouble(const std::string& valueStr, double min = std::numeric_limits<double>::min(), double max = std::numeric_limits<double>::max())
	{
		double value = atof(valueStr.c_str());
		value = value < min ? min : value;
		value = value > max ? max : value;
		return value;

	}

	int StrToInt(const std::string& valueStr, int min, int max)
	{
		int value = atoi(valueStr.c_str());
		value = value < min ? min : value;
		value = value > max ? max : value;
		return value;
	}

	Expression ExpressionFormStr(const std::string& expressionStr)
	{
		if (expressionStr == "is_more" || expressionStr == ">") {
			return Expression::is_more;
		}
		if (expressionStr == "is_more_or_equal" || expressionStr == ">=") {
			return Expression::is_more_or_equal;
		}
		else if (expressionStr == "is_less_or_equal" || expressionStr == "<=") {
			return Expression::is_more_or_equal;
		}
		else if (expressionStr == "is_less" || expressionStr == "<") {
			return Expression::is_less;
		}
		else if (expressionStr == "is_equal" || expressionStr == "==") {
			return Expression::is_equal;
		}
		else if (expressionStr == "is_not_equal" || expressionStr == "!=") {
			return Expression::is_not_equal;
		}

		return Expression::is_error;
	}

	Operation OperationFormStr(const std::string& expressionStr)
	{
		if (expressionStr == "+") {
			return Operation::is_addition;
		}
		if (expressionStr == "-") {
			return Operation::is_subtraction;
		}
		else if (expressionStr == "*") {
			return Operation::is_division;
		}
		else if (expressionStr == "/") {
			return Operation::is_division;
		}
		/*else if (expressionStr == "%") {
			return Operation::is_remainder;
		}*/

		return Operation::is_error;
	}

	// ///////////////////////////////////////////////////////////////////////////////////////

	bool count_bodies(const std::string& expressions, int number)
	{
		int countBodies = MySystem::currentSpace ? MySystem::currentSpace->_bodies.size() : 0;

		if (expressions == "is_more" || expressions == ">") {
			return countBodies > number;
		}
		if (expressions == "is_more_or_equal" || expressions == ">=") {
			return countBodies >= number;
		}
		else if (expressions == "is_less" || expressions == "<") {
			return countBodies < number;
		}
		else if (expressions == "is_less_or_equal" || expressions == "<=") {
			return countBodies <= number;
		}
		else if (expressions == "is_equal" || expressions == "==") {
			return countBodies == number;
		}

		return false;
	}

	bool max_speed_body(const std::string& expressions, float speed)
	{
		float maxSpeed = MySystem::currentSpace ? MySystem::currentSpace->GetMaxSpeed().length() : 0;
		help::Log("MAX_SPEED: " + std::to_string(maxSpeed) + " > " + std::to_string(speed));

		if (expressions == "is_more" || expressions == ">") {
			return maxSpeed > speed;
		}
		if (expressions == "is_more_or_equal" || expressions == ">=") {
			return maxSpeed >= speed;
		}
		else if (expressions == "is_less" || expressions == "<") {
			return maxSpeed < speed;
		}
		else if (expressions == "is_less_or_equal" || expressions == "<=") {
			return maxSpeed <= speed;
		}
		else if (expressions == "is_equal" || expressions == "==") {
			return maxSpeed == speed;
		}

		return false;
	}

	void RunCommandIf(const std::string& questNameLeft, const std::string& paramLeft,
						const std::string& expressionStr,
						const std::string& questNameRight, const std::string& paramRight,
						const std::string& questName, const std::string& commandName)
	{
		double valuLeft = 0.0;
		double valuRight = 0.0;

		// TODO:
		// LEFT
		if (Quest::Ptr questPtr = QuestManager::GetQuest(questNameLeft)) {
			auto it = questPtr->_params.find(paramLeft);
			if (it != questPtr->_params.end()) {
				valuLeft = StrToDouble((it->second));
			}
		}
		else {
			auto it = Quest::globalParams.find(paramLeft);
			if (it != Quest::globalParams.end()) {
				valuLeft = StrToDouble((it->second));
			}
			else {
				valuLeft = StrToDouble(paramLeft);
			}
		}

		// RIGHT
		if (Quest::Ptr questPtr = QuestManager::GetQuest(questNameRight)) {
			auto it = questPtr->_params.find(paramRight);
			if (it != questPtr->_params.end()) {
				valuRight = StrToDouble((it->second));
			}
		}
		else {
			auto it = Quest::globalParams.find(paramRight);
			if (it != Quest::globalParams.end()) {
				valuRight = StrToDouble((it->second));
			}
			else {
				valuLeft = StrToDouble(paramRight);
			}
		}

		Expression expression = ExpressionFormStr(expressionStr);
		bool needRun = false;

		switch (expression)
		{
		case quest::Expression::is_error:
			// TODO: ASSERT
			break;
		case quest::Expression::is_more:
			needRun = valuLeft > valuRight;
			break;
		case quest::Expression::is_more_or_equal:
			needRun = valuLeft >= valuRight;
			break;
		case quest::Expression::is_less:
			needRun = valuLeft < valuRight;
			break;
		case quest::Expression::is_less_or_equal:
			needRun = valuLeft <= valuRight;
			break;
		case quest::Expression::is_equal:
			needRun = valuLeft == valuRight;
			break;
		case quest::Expression::is_not_equal:
			needRun = valuLeft != valuRight;
			break;
		default:
			break;
		}

		if (needRun) {
			Commands* commandsPtr = nullptr;

			if (Quest::Ptr questPtr = QuestManager::GetQuest(questName)) {
				auto it = questPtr->_commandMap.find(commandName);
				if (it != questPtr->_commandMap.end()) {
					commandsPtr = &it->second;
				}
			}
			else {
				auto it = Quest::globalCommandsMap.find(commandName);
				if (it != Quest::globalCommandsMap.end()) {
					commandsPtr = &it->second;
				}
			}

			if (commandsPtr) {
				CommandManager::Run(*commandsPtr);
			}
		}
	}

	void ValueOperation(const std::string& questNameLeft, const std::string& paramLeft,
					const std::string& operationStr,
					const std::string& questNameRight, const std::string& paramRight,
					const std::string& questNameResult, const std::string& paramResult)
	{
		double valuLeft = 0.0;
		double valuRight = 0.0;
		double valuResult = 0.0;

		// TODO:
		// LEFT
		if (Quest::Ptr questPtr = QuestManager::GetQuest(questNameLeft)) {
			auto it = questPtr->_params.find(paramLeft);
			if (it != questPtr->_params.end()) {
				valuLeft = StrToDouble((it->second));
			}
		}
		else {
			auto it = Quest::globalParams.find(paramLeft);
			if (it != Quest::globalParams.end()) {
				valuLeft = StrToDouble((it->second));
			}
			else {
				valuLeft = StrToDouble(paramLeft);
			}
		}

		// RIGHT
		if (Quest::Ptr questPtr = QuestManager::GetQuest(questNameRight)) {
			auto it = questPtr->_params.find(paramRight);
			if (it != questPtr->_params.end()) {
				valuRight = StrToDouble((it->second));
			}
		}
		else {
			auto it = Quest::globalParams.find(paramRight);
			if (it != Quest::globalParams.end()) {
				valuRight = StrToDouble((it->second));
			}
			else {
				valuLeft = StrToDouble(paramRight);
			}
		}

		// RESULT
		if (Quest::Ptr questPtr = QuestManager::GetQuest(questNameResult)) {
			auto it = questPtr->_params.find(paramResult);
			if (it != questPtr->_params.end()) {
				valuResult = StrToDouble((it->second));
			}
		}
		else {
			auto it = Quest::globalParams.find(paramResult);
			if (it != Quest::globalParams.end()) {
				valuResult = StrToDouble((it->second));
			}
			else {

				// ASSERT
				return;
			}
		}

		Operation operation = OperationFormStr(operationStr);
		bool hasResult = false;

		switch (operation)
		{
		case quest::Operation::is_error:
			// TODO: ASSERT
			break;
		case quest::Operation::is_addition:
			valuResult = valuLeft + valuRight;
			hasResult = true;
			break;
		case quest::Operation::is_subtraction:
			valuResult = valuLeft - valuRight;
			hasResult = true;
			break;
		case quest::Operation::is_division:
			valuResult = valuLeft / valuRight;
			hasResult = true;
			break;
		case quest::Operation::is_multiplication:
			valuResult = valuLeft * valuRight;
			hasResult = true;
			break;
		/*case quest::Operation::is_remainder:
			break;*/
		default:
			break;
		}

		if (hasResult) {
			// RESULT
			if (Quest::Ptr questPtr = QuestManager::GetQuest(questNameResult)) {
				auto it = questPtr->_params.find(paramResult);
				if (it != questPtr->_params.end()) {
					it->second = std::to_string(valuResult);
					return;
				}
			}
			else {
				auto it = Quest::globalParams.find(paramResult);
				if (it != Quest::globalParams.end()) {
					it->second = std::to_string(valuResult);
					return;
				}
			}
		}
	}
}
