#pragma once

#include <string>
#include <map>
#include <MyStl/shared.h>


class Quest {
public:
	using Ptr = mystd::shared<Quest>;

	enum class State {
		NONE,
		INACTINE,
		ACTINE,
		REVARD,
		COMPLETE
	};

public:
	Quest() = delete;
	Quest(const std::string& name): _name(name) {}

	virtual ~Quest() = default;

	virtual void Init() = 0;
	virtual void Update() {};

	const std::string& Name() { return _name; };
	void SetState(State state) {
		_state = state;
		Init();
	}

	State GetState() { return _state; }

private:
	const std::string _name;
	State _state = State::INACTINE;
};
