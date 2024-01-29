
#include "SpaceManagerUI.h"
#include "MySystem/MySystem.h"
#include "../Objects/SpaceManager.h"

#include <cmath>
#include "imgui.h"
#include <glm/ext/scalar_constants.hpp>
#include "CommonData.h"
#include "Math/Vector.h"
#include "../Objects/Body.h"
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

    if (!_mySystem || !_mySystem->_space) {
        return;
    }

    Space* currentSpacePtr = _mySystem->_space.get();
    std::string currentClass = currentSpacePtr->GetNameClass();
    ImGui::Text(currentClass.c_str());

    ImGui::Separator();
    if (ImGui::Button("Save", { 128.f, 32.f })) {
        if (_mySystem && _mySystem->_space) {
            _mySystem->_space->Save();
        }
    }
    
    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Reload", { 128.f, 32.f })) {
        if (_mySystem && _mySystem->_space) {
            _mySystem->_space->Load();
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Correct system", { 128.f, 32.f })) {
        _mySystem->_space->RemoveVelocity(true);
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Clear", { 128.f, 32.f })) {
        if (_mySystem && _mySystem->_space) {
            Space& space = *_mySystem->_space;

            Body::Ptr heaviestBody = space.GetHeaviestBody();
            space._bodies.clear();
            space._bodies.emplace_back(heaviestBody);
            space.Preparation();

        }
    }

    ImGui::Separator();

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate round", { 128.f, 32.f })) {
        if (_mySystem && _mySystem->_space) {
            if (_mySystem && _mySystem->_space) {
                if (_mySystem && _mySystem->_space->_selectBody == nullptr && _mySystem && _mySystem->_space->_bodies.size() == 1) {
                    _mySystem->_space->_selectBody = _mySystem->_space->_bodies.front();
                }
                int count = 1000;
                std::string countStr = _mySystem->_space->_params["COUNT"];
                if (!countStr.empty()) {
                    count = std::stoi(countStr);
                }

                _mySystem->_space->_bodies.reserve(count);

                double minSpaceRange = 1000;
                double spaceRange = 5000;
                Math::Vector3d pos;

                int i = 0;
                while (i < count) {
                    pos.x = help::random(-spaceRange, spaceRange);
                    pos.y = help::random(-spaceRange, spaceRange);
                    pos.z = help::random(-1.f, 1.f);

                    double radius = pos.length();

                    if (radius > spaceRange) {
                        continue;
                    }

                    if (radius < minSpaceRange) {
                        continue;
                    }
                    ++i;

                    SpaceManager::AddObjectOnOrbit(_mySystem->_space.get(), pos, false);
                }

                _mySystem->_space->Preparation();
            }
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate sphere", { 128.f, 32.f })) {
        if (_mySystem && _mySystem->_space) {
            if (_mySystem && _mySystem->_space) {
                if (_mySystem && _mySystem->_space->_selectBody == nullptr && _mySystem && _mySystem->_space->_bodies.size() == 1) {
                    _mySystem->_space->_selectBody = _mySystem->_space->_bodies.front();
                }
                int count = 1000;
                float minSpaceRange = 1000;
                float spaceRange = 5000;

                std::string countStr = _mySystem->_space->_params["COUNT"];
                if (!countStr.empty()) {
                    count = std::stoi(countStr);
                }

                std::string minSpaceRangeStr = _mySystem->_space->_params["MIN_RADIUS"];
                if (!minSpaceRangeStr.empty()) {
                    minSpaceRange = std::stoi(minSpaceRangeStr);
                }

                std::string spaceRangeStr = _mySystem->_space->_params["MAX_RADIUS"];
                if (!spaceRangeStr.empty()) {
                    spaceRange = std::stoi(spaceRangeStr);
                }

                _mySystem->_space->_bodies.reserve(count);
                Math::Vector3d pos;

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

                    SpaceManager::AddObjectOnOrbit(_mySystem->_space.get(), pos, false);
                }

                _mySystem->_space->Preparation();
            }
        }
    }

    ImGui::Separator();
}
