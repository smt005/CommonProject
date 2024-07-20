// ◦ Xyz ◦
#pragma once

#include "Space.h"
#include <Math/Vector.h>
#include <Object/Color.h>

class MySystem;
class Space;

class SpaceManager
{
public:
	static void AddObjectOnOrbit(Space* space, Math::Vector3& pos, bool withAssotiation = true);
	static void AddObjectDirect(Space* space, Math::Vector3& pos, Math::Vector3& vel);
	static void AddObjects(Space* space, int count, double spaceRange, double conventionalMass = -1);
	static void AddObject(const std::string& nameModel, const Math::Vector3& pos, const Math::Vector3& vel, float mass, const Color& color = Color(1.f, 1.f, 1.f, 1.f));

	static unsigned int SetView(MySystem* systemMy);

	static void Save(Space::Ptr space);
	static Space::Ptr Load(const std::string& name);

	static const std::vector<std::string>& GetListClasses();
	static std::shared_ptr<Space> CopySpace(const std::string& className, Space* space);
};
