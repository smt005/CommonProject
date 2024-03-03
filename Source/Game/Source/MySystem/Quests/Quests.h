#pragma once
#include "Quest.h"

class QuestStart final : public Quest {
public:
	QuestStart(const std::string& name): Quest(name) {}
	void Activete() override;
};

class QuestSphere100 final : public Quest {
public:
	QuestSphere100(const std::string& name) : Quest(name) {}
	void Activete() override;
};

class QuestSphere final : public Quest {
public:
	QuestSphere(const std::string& name) : Quest(name) {}
	void Activete() override;
};
