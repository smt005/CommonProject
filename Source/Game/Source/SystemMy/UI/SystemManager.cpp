
#include "SystemManager.h"
#include "../Objects/SpaceManager.h"

#include <cmath>
#include "imgui.h"
#include <glm/ext/scalar_constants.hpp>
#include "CommonData.h"
#include "../SystemMy.h"
#include "Math/Vector.h"
#include "../Objects/SystemClass.h"
#include "../Objects/SystemMapEasyMerger.h"
#include "../Objects/SystemMapShared.h"
#include "../Objects/SystemMapMyShared.h"
#include "../SaveManager.h"
#include "Math/Vector.h"

SystemManager::SystemManager() : UI::Window(this) {
    SetId("SystemManager");
    Close();
}

SystemManager::SystemManager(SystemMy* systemMy)
    : UI::Window(this)
    , _systemMy(systemMy)
{
    SetId("SystemManager");
}

void SystemManager::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize);

    _y = 100.f;

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });

    ++CommonData::lockAction;
}

void SystemManager::OnClose() {
    --CommonData::lockAction;
}

void SystemManager::Update() {

}

void SystemManager::Draw() {
    //ImGuiStyle& style = ImGui::GetStyle();
    //style.FramePadding.y = 3.f;

    if (ImGui::Button("Save", { 128.f, 32.f })) {
        if (_systemMy && _systemMy->_systemMap) {
            _systemMy->_systemMap->Save();
        }
    }
    
    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Reload", { 128.f, 32.f })) {
        if (_systemMy && _systemMy->_systemMap) {
            _systemMy->_systemMap->Load();
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Correct system", { 128.f, 32.f })) {
        _systemMy->_systemMap->RemoveVelocity(true);
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Clear", { 128.f, 32.f })) {
        if (_systemMy && _systemMy->_systemMap) {
            SystemMap& systemMap = *_systemMy->_systemMap;

#if SYSTEM_MAP < 7
            Body* heaviestBody = &systemMap.RefFocusBody();
            std::vector<Body*> bodies;
            bodies.emplace_back(heaviestBody);

            for (auto* bodyPtr : systemMap._bodies) {
                if (bodyPtr != heaviestBody) {
                    delete bodyPtr;
                }
            }

            std::swap(systemMap._bodies, bodies);
            _systemMy->_systemMap->DataAssociation();
#else
            Body::Ptr heaviestBody = systemMap.GetHeaviestBody();
            systemMap._bodies.clear();
            systemMap._bodies.emplace_back(heaviestBody);
            _systemMy->_systemMap->DataAssociation();
#endif
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate 1", { 128.f, 32.f })) {
        if (_systemMy && _systemMy->_systemMap) {
            SystemMap& systemMap = *_systemMy->_systemMap;

            Body* star = &systemMap.RefFocusBody();
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

                        velocity *= std::sqrtf(systemMap._constGravity * starMass / glm::length(gravityVector));
                        systemMap.Add("BrownStone", pos, velocity, mass, "");
                    }
                }
            }
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate 2", { 128.f, 32.f })) {
        if (_systemMy && _systemMy->_systemMap) {
            if (_systemMy && _systemMy->_systemMap) {
                SystemMap& systemMap = *_systemMy->_systemMap;

                int count = 333;
                double spaceRange = 10000;
                Math::Vector3d pos;

                for (int i = 0; i < count; ++i) {                    
                    pos.x = help::random(-spaceRange, spaceRange);
                    pos.y = help::random(-spaceRange, spaceRange);
                    pos.z = 0;// help::random(-10, 10);
                    
                    float mass = help::random(50, 150);

                    SpaceManager::AddObjectOnOrbit(_systemMy->_systemMap.get(), pos);
                }
            }
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate 3", { 128.f, 32.f })) {
        if (_systemMy && _systemMy->_systemMap) {
            SystemMap& systemMap = *_systemMy->_systemMap;
            bool hasStar = systemMap._selectBody;

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

                    SpaceManager::AddObjectOnOrbit(_systemMy->_systemMap.get(), pos);
                }
            }
        }
    }
}

void SystemManager::CreateOrbitBody(double x, double y, double z) {
    if (!_systemMy || !_systemMy->_systemMap) {
        return;
    }

    SystemMap& systemMap = *_systemMy->_systemMap;

    systemMap.Add("BrownStone", Math::Vector3d(x, y, z), Math::Vector3d(), 100, "");
}

void SystemManager::CreateOrbitBody(double x, double y, double z, double starMass, double startX, double startY, double startZ) {
    if (!_systemMy || !_systemMy->_systemMap) {
        return;
    }
       
    SystemMap& systemMap = *_systemMy->_systemMap;

    glm::vec3 pos(x, y, z);
    glm::vec3 starPos(startX, startY, startZ);
    float mass = 100.f;

    glm::vec3 gravityVector = pos - starPos;
    glm::vec3 normalizeGravityVector = glm::normalize(gravityVector);

    float g90 = glm::pi<float>() / 2.0;
    glm::vec3 velocity(normalizeGravityVector.x * std::cos(g90) - normalizeGravityVector.y * std::sin(g90),
        normalizeGravityVector.x * std::sin(g90) + normalizeGravityVector.y * std::cos(g90),
        0.f);

    velocity *= std::sqrtf(systemMap._constGravity * starMass / glm::length(gravityVector));
    systemMap.Add("BrownStone", pos, velocity, mass, "");
}
