// ◦ Xyz ◦
#include "QuestCondition.h"
#include <MySystem/MySystem.h>
#include "../Commands/Commands.h"
#include "../Objects/Space.h"
#include "Common/Help.h"

namespace quest
{
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
}
