// ◦ Xyz ◦
#include "QuestCondition.h"
#include <MySystem/MySystem.h>
#include "../Commands/Commands.h"
#include "../Objects/Space.h"

namespace quest {
	bool count_boties(const std::string& expressions, int number) {
		int countBodies = MySystem::currentSpace ? MySystem::currentSpace->_bodies.size() : 0;

		if (expressions == "is_more") {
			return countBodies > number;
		}
		else if(expressions == "is_less") {
			return countBodies < number;
		}
		else if (expressions == "is_equal") {
			return countBodies == number;
		}

		return false;
	}
}
