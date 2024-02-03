#pragma once

#include <string>
#include "MyStl/shared.h"

class Body {
public:
	using Ptr = mystd::shared<Body>;

	Body() = default;
	virtual ~Body() {
		delete[] _name;
	}

	void SetName(const std::string& name) {
		size_t size = name.size();
		_name = new char[size + 1];
		memcpy(_name, name.data(), size);
		_name[size] = '\0';
	}

	const char* const Name() const {
		return _name;
	}

	virtual Math::Vector3d GetPos() const { return Math::Vector3d(); }
	virtual void SetPos(const Math::Vector3d& pos) { }
	virtual void SetVelocity(const Math::Vector3d& velocity) { }
	virtual bool GetModel() const { return false; }
	virtual Model& getModel() { static Model def; return def; }
	virtual const glm::mat4x4& getMatrix() const { static glm::mat4x4 def; return def; }
	virtual Math::Vector3 PosOnScreen(const glm::mat4x4& matCamera, bool applySizeScreen) { return Math::Vector3(); }
	virtual bool hit(const glm::mat4x4& matCamera) { return false; }
	virtual void Rotate() {}
	virtual void Scale() {}

public:
	bool visible = true;
	char* _name = nullptr;
};
