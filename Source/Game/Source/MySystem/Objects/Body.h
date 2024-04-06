// ◦ Xyz ◦
#pragma once

#include <string>
#include <memory>
#include <glm/mat4x4.hpp>
#include "MyStl/shared.h"
#include "Math/Vector.h"
#include "../../Engine/Source/Object/Model.h"

class Model;

class Body {
public:
	using Ptr = mystd::shared<Body>;
	//using Ptr = std::shared_ptr<Body>;

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

	virtual Math::Vector3 GetPos() const { return Math::Vector3(); }
	virtual void SetPos(const Math::Vector3& pos) { }
	virtual void SetVelocity(const Math::Vector3& velocity) { }
	virtual bool HasModel() const { return false; }
	virtual Model& getModel() { static Model def; return def; }
	virtual const glm::mat4x4& getMatrix() const { static glm::mat4x4 def; return def; }
	virtual Math::Vector3 PosOnScreen(const glm::mat4x4& matCamera, bool applySizeScreen) { return Math::Vector3(); }
	virtual bool hit(const glm::mat4x4& matCamera) { return false; }
	virtual void Rotate() {}
	virtual void CalcScale() {}

	virtual Math::Vector3& Velocity() { static Math::Vector3 def; return def; }
	virtual Math::Vector3& Force() { static Math::Vector3 def; return def; }
	virtual float& Mass() { static float def = 0; return def; }
	virtual float& Scale() { static float def = 0; return def; }

public:
	bool visible = true;
	char* _name = nullptr;
};
