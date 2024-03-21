#pragma once

#include <string>
#include <vector>
#include <functional>
#include "MyStl/Singleton.h"

class Event : public mystd::Singleton<Event> {
public:
	Event() = default;
	virtual ~Event() = default;

	void Action() {
		for (auto& pairObserver : _observers) {
			pairObserver.second();
		}
	}

	void Add(const std::string& id, std::function<void()> function) {
		if (_observers.end() != std::find_if(_observers.begin(), _observers.end(), [&id](const std::pair<std::string, std::function<void()>>& pair) {
			return pair.first == id;
		})) {
			return;
		}
		_observers.emplace_back(id, function);
	}

	void Remove(const std::string& id) {
		auto it = std::find_if(_observers.begin(), _observers.end(), [&id](const std::pair<std::string, std::function<void()>>& pair) {
			return pair.first == id;
		});
		if (it == _observers.end()) {
			return;
		}
		_observers.erase(it);
	}

private:
	std::vector<std::pair<std::string, std::function<void()>>> _observers;
};
