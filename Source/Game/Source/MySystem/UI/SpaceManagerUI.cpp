
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
#include "Math/Vector.h"

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

    _y = 100.f;

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
    //ImGuiStyle& style = ImGui::GetStyle();
    //style.FramePadding.y = 3.f;

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
            space.DataAssociation();

        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate 1", { 128.f, 32.f })) {
        if (_mySystem && _mySystem->_space) {
            Space& space = *_mySystem->_space;

            Body* star = &space.RefFocusBody();
            if (star) {
                auto starPosT = star->GetPos();
                glm::vec3 starPos = glm::vec3(starPosT.x, starPosT.y, starPosT.z);
                float starMass = star->_mass;

                float dist = 1000.0f;
                int countX = 10; // 15;
                int countY = 10; // 15;

                for (int iX = -countX; iX < countX; ++iX) {
                    for (int iY = -countY; iY < countY; ++iY) {
                        if (iX == 0 && iY == 0) {
                            continue;
                        }

                        glm::vec3 pos(iX * dist, iY * dist, 0);
                        //pos.z += help::random(-1000.f, 1000.f);

                        float mass = 100.f;

                        glm::vec3 gravityVector = pos - starPos;
                        glm::vec3 normalizeGravityVector = glm::normalize(gravityVector);

                        float g90 = glm::pi<float>() / 2.0;
                        glm::vec3 velocity(normalizeGravityVector.x * std::cos(g90) - normalizeGravityVector.y * std::sin(g90),
                            normalizeGravityVector.x * std::sin(g90) + normalizeGravityVector.y * std::cos(g90),
                            0.f);

                        velocity *= std::sqrtf(space._constGravity * starMass / glm::length(gravityVector));
                        space.Add("BrownStone", pos, velocity, mass, "");
                    }
                }
            }
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate 2", { 128.f, 32.f })) {
        if (_mySystem && _mySystem->_space) {
            if (_mySystem && _mySystem->_space) {
                int count = 333;
                double spaceRange = 10000;
                Math::Vector3d pos;

                for (int i = 0; i < count; ++i) {                    
                    pos.x = help::random(-spaceRange, spaceRange);
                    pos.y = help::random(-spaceRange, spaceRange);
                    pos.z = 0;// help::random(-10, 10);
                    
                    float mass = help::random(50, 150);

                    SpaceManager::AddObjectOnOrbit(_mySystem->_space.get(), pos);
                }
            }
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate 3", { 128.f, 32.f })) {
        if (_mySystem && _mySystem->_space) {
            Space& space = *_mySystem->_space;
            bool hasStar = space._selectBody;

            float dist = 1000.0f;
            int countX = 10; // 15;
            int countY = 10; // 15;

            for (int iX = -countX; iX < countX; ++iX) {
                for (int iY = -countY; iY < countY; ++iY) {
                    if (hasStar && (iX == 0 && iY == 0)) {
                        continue;
                    }
                    double iZ = help::random(-100.f, 100.f);

                    Math::Vector3d pos((double)iX * dist, (double)iY * dist, (double)iZ);

                    float mass = 100.f;

                    SpaceManager::AddObjectOnOrbit(_mySystem->_space.get(), pos);
                }
            }
        }
    }
}

void SpaceManagerUI::CreateOrbitBody(double x, double y, double z) {
    if (!_mySystem || !_mySystem->_space) {
        return;
    }

    Space& space = *_mySystem->_space;

    space.Add("BrownStone", Math::Vector3d(x, y, z), Math::Vector3d(), 100, "");
}

void SpaceManagerUI::CreateOrbitBody(double x, double y, double z, double starMass, double startX, double startY, double startZ) {
    if (!_mySystem || !_mySystem->_space) {
        return;
    }
       
    Space& space = *_mySystem->_space;

    glm::vec3 pos(x, y, z);
    glm::vec3 starPos(startX, startY, startZ);
    float mass = 100.f;

    glm::vec3 gravityVector = pos - starPos;
    glm::vec3 normalizeGravityVector = glm::normalize(gravityVector);

    float g90 = glm::pi<float>() / 2.0;
    glm::vec3 velocity(normalizeGravityVector.x * std::cos(g90) - normalizeGravityVector.y * std::sin(g90),
        normalizeGravityVector.x * std::sin(g90) + normalizeGravityVector.y * std::cos(g90),
        0.f);

    velocity *= std::sqrtf(space._constGravity * starMass / glm::length(gravityVector));
    space.Add("BrownStone", pos, velocity, mass, "");
}
