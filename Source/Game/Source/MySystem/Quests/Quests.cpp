#include "Quests.h"
#include "MySystem/MySystem.h"
#include "../Objects/Space.h"
#include "../Objects/SpaceManager.h"
#include "Common/Help.h"
#include "../../CUDA/Source/Wrapper.h"

// QuestStart
void QuestStart::Activete() {
	MySystem::currentSpace.reset();
    MySystem::currentSpace = SpaceManager::Load("MAIN");
}

// QuestSphere100
void QuestSphere100::Activete() {
    if (!MySystem::currentSpace) {
        MySystem::currentSpace = SpaceManager::Load("MAIN");
    }

    //...
    int count = 1000;
    float minSpaceRange = 100;
    float spaceRange = 5000;

    MySystem::currentSpace->_bodies.reserve(count);
    Math::Vector3 pos;

    int i = 0;
    while (i < count) {
        pos.x = help::random(-spaceRange, spaceRange);
        pos.y = help::random(-spaceRange, spaceRange);
        pos.z = help::random(-spaceRange, spaceRange);

        double radius = pos.length();

        if (radius > spaceRange) {
            continue;
        }

        if (radius < minSpaceRange) {
            continue;
        }
        ++i;

        SpaceManager::AddObjectOnOrbit(MySystem::currentSpace.get(), pos, false);
    }

    MySystem::currentSpace->Preparation();
}

// QuestSphere
void QuestSphere::Activete() {
	if (!MySystem::currentSpace) {
		MySystem::currentSpace = SpaceManager::Load("MAIN");
	}

    // TODO:
    CUDA::multithread = true;
    
    //...
    int count = 1000;
    float minSpaceRange = 1000;
    float spaceRange = 5000;

    std::string countStr = MySystem::currentSpace->_params["COUNT"];
    if (!countStr.empty()) {
        count = std::stoi(countStr);
    }

    std::string minSpaceRangeStr = MySystem::currentSpace->_params["MIN_RADIUS"];
    if (!minSpaceRangeStr.empty()) {
        minSpaceRange = std::stoi(minSpaceRangeStr);
    }

    std::string spaceRangeStr = MySystem::currentSpace->_params["MAX_RADIUS"];
    if (!spaceRangeStr.empty()) {
        spaceRange = std::stoi(spaceRangeStr);
    }

    MySystem::currentSpace->_bodies.reserve(count);
    Math::Vector3 pos;

    int i = 0;
    while (i < count) {
        pos.x = help::random(-spaceRange, spaceRange);
        pos.y = help::random(-spaceRange, spaceRange);
        pos.z = help::random(-spaceRange, spaceRange);

        double radius = pos.length();

        if (radius > spaceRange) {
            continue;
        }

        if (radius < minSpaceRange) {
            continue;
        }
        ++i;

        SpaceManager::AddObjectOnOrbit(MySystem::currentSpace.get(), pos, false);
    }

    MySystem::currentSpace->Preparation();
}
