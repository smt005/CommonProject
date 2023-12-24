#pragma once

#include "Space.h"
#include "Math/Vector.h"

class MySystem;
class Space;

class SpaceManager {
public:
	static void AddObjectOnOrbit(Space* space, Math::Vector3d& pos);
	static void AddObjectDirect(Space* space, Math::Vector3d& pos, Math::Vector3d& vel);

	static unsigned int SetView(MySystem* systemMy);

	static void Save(Space::Ptr space);
	static Space::Ptr Load(const std::string& name);
};
