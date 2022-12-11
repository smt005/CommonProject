
#pragma once

#include "Object/Map.h"
#include "Object/Object.h"

namespace Engine { class Callback; }
typedef std::shared_ptr<Engine::Callback> CallbackPtr;

class MenuMap final : public Map {
private:
	class Puck : public Object {
	public:
		Puck(const string& name, const string& modelName, const vec3& pos = vec3(0.0f)) : Object(name, modelName, pos) {}
		void action() override;

	private:
		float _force = -1.0f;
	};

	class Target : public Object {
	public:
		Target(const string& name, const string& modelName, const vec3& pos = vec3(0.0f)) : Object(name, modelName, pos) {}
		void action() override;

	private:
		float _angle = 0.0f;
		float _dist = 150;
		float _dAngle = 0.01f;
		float _force = -0.5f;
	};

// MenuMap
public:
	MenuMap();
	~MenuMap() {
		target.reset();
	}
	
	bool create(const string& name) override;
	void action() override;

	void hit(const int x, const int y);

private:
	int _tact = 0;
	CallbackPtr _callbackPtr;

private:
	static Target::Ptr target;
	static bool enableForce;
};

// MenuOrtoMap
class MenuOrtoMap final : public Map {
public:
	bool create(const string& name) override;
};
