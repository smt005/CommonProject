
#include "SpaceManagerUI.h"
#include "MySystem/MySystem.h"
#include "../Objects/SpaceManager.h"

#include <cmath>
#include "imgui.h"
#include <glm/ext/scalar_constants.hpp>
#include "CommonData.h"
#include "Math/Vector.h"
#include "../Objects/BodyData.h"
#include "../Objects/Space.h"
#include "../Objects/Space.h"
#include <../../CUDA/Source/Emulate.h>

namespace {
    Space::Ptr spaceTemp;
}

SpaceManagerUI::SpaceManagerUI() : UI::Window(this) {
    SetId("SpaceManager");
    Close();
}

SpaceManagerUI::SpaceManagerUI(MySystem* mySystem)
    : UI::Window(this)
    , _mySystem(mySystem)
{}

void SpaceManagerUI::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize);

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });

    ++CommonData::lockAction;
}

void SpaceManagerUI::OnClose() {
    --CommonData::lockAction;
}

void SpaceManagerUI::Update() {

}

void SpaceManagerUI::Draw() {
    //spaceTemp.reset();

    //ImGuiStyle& style = ImGui::GetStyle();
    //style.FramePadding.y = 3.f;

    if (!MySystem::currentSpace) {
        return;
    }

    Space* currentSpacePtr = MySystem::currentSpace.get();
    std::string currentClass = currentSpacePtr->GetNameClass();
    ImGui::Text(currentClass.c_str());

    ImGui::Separator();
    if (ImGui::Button("Save", { 128.f, 32.f })) {
        if (_mySystem && MySystem::currentSpace) {
            MySystem::currentSpace->Save();
        }
    }
    
    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Reload", { 128.f, 32.f })) {
        if (_mySystem && MySystem::currentSpace) {
            MySystem::currentSpace->Load();
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Correct system", { 128.f, 32.f })) {
        MySystem::currentSpace->RemoveVelocity(true);
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Clear", { 128.f, 32.f })) {
        if (_mySystem && MySystem::currentSpace) {
            Space& space = *MySystem::currentSpace;

            auto heaviestBody = space.GetHeaviestBody();
            space._bodies.clear();
            space._bodies.emplace_back(heaviestBody);
            space.Preparation();

        }
    }

    ImGui::Separator();

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate round", { 128.f, 32.f })) {
        if (_mySystem && MySystem::currentSpace) {
            if (_mySystem && MySystem::currentSpace) {
                if (_mySystem && MySystem::currentSpace->_selectBody == nullptr && _mySystem && MySystem::currentSpace->_bodies.size() == 1) {
                    MySystem::currentSpace->_selectBody = MySystem::currentSpace->_bodies.front();
                }

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
                    pos.z = 0.f; //pos.z = help::random(-1.f, 1.f);

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
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate sphere", { 128.f, 32.f })) {
        if (_mySystem && MySystem::currentSpace) {
            if (_mySystem && MySystem::currentSpace) {
                if (_mySystem && MySystem::currentSpace->_selectBody == nullptr && _mySystem && MySystem::currentSpace->_bodies.size() == 1) {
                    MySystem::currentSpace->_selectBody = MySystem::currentSpace->_bodies.front();
                }
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
        }
    }

    ImGui::Separator();
}
