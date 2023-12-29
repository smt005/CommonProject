#pragma once

#include <string>
#include <map>
#include <MyStl/shared.h>


class Quest {
public:
	using Ptr = mystd::shared<Quest>;

public:
	Quest() = delete;
	Quest(const std::string& name);

public:
	static bool Load();

private:
	const std::string _name;
	std::string _test;

private:
	static std::map<std::string, Quest::Ptr> quests;
};
