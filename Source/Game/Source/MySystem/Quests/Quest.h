// ◦ Xyz ◦
#pragma once

#include <string>
#include <map>
#include <functional>
#include <MyStl/shared.h>
#include "../Commands/Commands.h"

class Quest
{
public:
	using Ptr = mystd::shared<Quest>;

	enum class State
	{
		NONE,
		DEACTIVE,
		ACTIVE,
		COMPLETE
	};

public:
	Quest() = delete;
	Quest(const std::string& name)
		: _name(name)
	{}

	virtual ~Quest() = default;

	virtual void Activete() {};
	virtual void Deactivation() {};
	virtual void Complete() {};
	virtual void Update() {};

	const std::string& Name() const { return _name; };
	void SetState(State state)
	{
		_state = state;
	}

	void ActivateState(State state = State::NONE)
	{
		if (state != State::NONE) {
			SetState(state);
		}

		if (_state == Quest::State::ACTIVE) {
			CommandManager::Run(_commandsOnInit);
			Activete();
		}
		else if (_state == Quest::State::DEACTIVE) {
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
	std::string _description;

	Commands _commandsOnInit;
	Commands _commandsOnTap;
	Commands _commandsOnUpdate;
	Commands _commandsOnCondition;
	Commands _commandsDebug;
	std::map<std::string, Commands> _commandMap;

public:
	static std::map<std::string, std::string> globalParams;
	static std::map<std::string, Commands> globalCommandsMap;
};
