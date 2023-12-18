#pragma once

#include "SystemClass.h"
#include "SystemMapMyShared.h"
#include "Math/Vector.h"

using Space = SystemMap;

class SystemMy;

class SpaceManager {
public:
	static void AddObjectOnOrbit(Space* space, Math::Vector3d& pos);
	static void AddObjectDirect(Space* space, Math::Vector3d& pos, Math::Vector3d& vel);

	static void SetView(SystemMy* systemMy);
};
