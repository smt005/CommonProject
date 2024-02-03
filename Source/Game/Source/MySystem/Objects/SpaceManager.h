#pragma once

#include "Space.h"
#include "Math/Vector.h"

class MySystem;
class Space;

class SpaceManager {
public:
	static void AddObjectOnOrbit(Space* space, Math::Vector3& pos, bool withAssotiation = true);
	static void AddObjectDirect(Space* space, Math::Vector3& pos, Math::Vector3& vel);
	static void AddObjects(Space* space, int count, double spaceRange, double conventionalMass = -1);

	static unsigned int SetView(MySystem* systemMy);

	static void Save(Space::Ptr space);
	static Space::Ptr Load(const std::string& name);
	
	static const std::vector<std::string>& GetListClasses();
	static std::shared_ptr<Space> CopySpace(const std::string& className, Space* space);

};
