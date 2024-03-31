// ◦ Xyz ◦
#pragma once

#include <string>
#include <map>
#include <functional>
#include <MyStl/shared.h>
#include "../Commands/Commands.h"

class Quest {
public:
	using Ptr = mystd::shared<Quest>;

	enum class State {
		NONE,
		DEACTIVE,
		ACTIVE,
		COMPLETE
	};

public:
	Quest() = delete;
	Quest(const std::string& name): _name(name) {}

	virtual ~Quest() = default;

	virtual void Activete() {};
	virtual void Deactivation() {};
	virtual void Complete() {};
	virtual void Update() {};

	const std::string& Name() const { return _name; };
	void SetState(State state) {
		_state = state;
	}

	void ActivateState(State state = State::NONE) {
		if (state != State::NONE) {
			SetState(state);
		}

		if (_state == Quest::State::ACTIVE) {
			CommandManager::Run(_commands);
			Activete();
		}
		else if (_state == Quest::State::DEACTIVE) {
			//CommandManager::Run(XXX);
			Deactivation();
		}
		else if (_state == Quest::State::COMPLETE) {
			Complete();
		}
	}

	State GetState() { return _state; }

public:
	std::string _name;
	State _state = State::DEACTIVE;
	std::map<std::string, std::string> _params;

	Commands _commands;
	Commands _commandsOnTap;
	Commands _commandsOnCondition;
};
