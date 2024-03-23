// ◦ Xyz ◦
#include "Quests.h"
#include "../Objects/Space.h"
#include "../Objects/SpaceManager.h"
#include <Common/Help.h>
#include <Wrapper.h>
#include <MyStl/Event.h>
#include  <MySystem/MySystem.h>

// QuestStart
void QuestStart::Activete() {
	MySystem::currentSpace.reset();
    MySystem::currentSpace = SpaceManager::Load("MAIN");


    std::string nextQuest = _params["next_quest"];

    if (!nextQuest.empty()) {
        if (_name == "Logo") {
            Event::Instance().Add(_name, [name = _name, nextQuest]() {
                if (MySystem::currentSpace) {
                    if (MySystem::currentSpace->_bodies.size() > 0) {
                        CommandManager::Run(Command("HideImage", { "Logo" }));
                        CommandManager::Run(Command("SetActiveQuest", { nextQuest, "ACTIVE" }));
                        Event::Instance().Remove(name);
                    }
                }
            });
        }
        else {
            int count = atoi(_params["count"].c_str());

            Event::Instance().Add(_name, [name = _name, nextQuest, count]() {
                if (MySystem::currentSpace) {
                    if (MySystem::currentSpace->_bodies.size() > count) {
                        CommandManager::Run(Command("SetActiveQuest", { nextQuest, "ACTIVE" }));
                        Event::Instance().Remove(name);
                    }
                }
            });
        }
    }
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

    MySystem::currentSpace->_bodies.reserve(count + 1);

    if (MySystem::currentSpace->_bodies.empty()) {
        SpaceManager::AddObject("BrownStone", Math::Vector3(0.f, 0.f, 0.f), Math::Vector3(0.f, 0.f, 0.f), 100.f);
        MySystem::currentSpace->_focusBody = MySystem::currentSpace->_bodies.front();
    }

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
    //CUDA::multithread = true;
    
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

    MySystem::currentSpace->_bodies.reserve(count + 1);

    if (MySystem::currentSpace->_bodies.empty()) {
        SpaceManager::AddObject("BrownStone", Math::Vector3(0.f, 0.f, 0.f), Math::Vector3(0.f, 0.f, 0.f), 100.f);
        MySystem::currentSpace->_focusBody = MySystem::currentSpace->_bodies.front();
    }

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
