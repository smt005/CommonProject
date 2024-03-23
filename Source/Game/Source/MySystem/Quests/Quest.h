// ◦ Xyz ◦
// ◦ Xyz ◦
#pragma once

#include <string>
#include <map>
#include <MyStl/shared.h>
#include "../Commands/Commands.h"

class Quest {
public:
	using Ptr = mystd::shared<Quest>;

	enum class State {
		NONE,
		INACTIVE,
		ACTIVE,
		COMPLETE
	};

public:
	Quest() = delete;
	Quest(const std::string& name): _name(name) {}

	virtual ~Quest() = default;

	virtual void Activete() {};
	virtual void Inactivete() {};
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
			Activete();
			CommandManager::Run(_commands);
		}
		else if (_state == Quest::State::INACTIVE) {
			Inactivete();
		}
		else if (_state == Quest::State::COMPLETE) {
			Complete();
		}
	}

	State GetState() { return _state; }

public:
	const std::string _name;
	State _state = State::INACTIVE;
	Commands _commands;
	std::map<std::string, std::string> _params;
};
