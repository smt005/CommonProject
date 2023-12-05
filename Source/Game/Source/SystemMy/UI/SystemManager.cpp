
#include "SystemManager.h"
#include <cmath>
#include "imgui.h"
#include <glm/ext/scalar_constants.hpp>
#include "../Objects/SystemMap.h"
#include "../SystemMy.h"
#include "Math/Vector.h"
#include "../Objects/SystemClass.h"
#include "../Objects/SystemMap.h"
#include "../Objects/SystemMapArr.h"
#include "../Objects/SystemMapStaticArr.h"
#include "../Objects/SystemMapDouble.h"
#include "../Objects/SystemMapEasyMerger.h"
#include "../SaveManager.h"

SystemManager::SystemManager() {
    SetId("SystemManager");
    Close();
}

SystemManager::SystemManager(SystemMy* systemMy)
    : UI::Window()
    , _systemMy(systemMy)
{
    SetId("SystemManager");
}

void SystemManager::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    _y = 100.f;

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });
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
        if (_systemMy && _systemMy->_systemMap) {
            SystemMap& systemMap = *_systemMy->_systemMap;

            if (Body* star = systemMap.GetBody("Sun")) {
                auto starPosT = star->GetPos();
                auto starVelT = star->_velocity;
                glm::vec3 starPos = glm::vec3(starPosT.x, starPosT.y, starPosT.z);
                glm::vec3 starVel = glm::vec3(starVelT.x, starVelT.y, starVelT.z);

                for (Body* body : systemMap.Objects()) {
                    auto posT = body->GetPos();
                    auto velT = body->_velocity;
                    glm::vec3 pos = glm::vec3(posT.x, posT.y, posT.z);
                    glm::vec3 vel = glm::vec3(velT.x, velT.y, velT.z);

                    pos -= starPos;
                    vel -= starVel;

                    body->SetPos(pos);
                    body->SetVelocity(vel);
                    
                }
            }
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Clear", { 128.f, 32.f })) {
        if (_systemMy && _systemMy->_systemMap) {
            SystemMap& systemMap = *_systemMy->_systemMap;

            std::vector<Body*> bodies;

            for (auto* bodyPtr : systemMap._bodies) {
                auto* namePtr = bodyPtr->Name();

                if (namePtr && std::string(namePtr) == "Sun") {
                    bodies.emplace_back(bodyPtr);
                } else {
                    delete bodyPtr;
                }
            }

            std::swap(systemMap._bodies, bodies);
            _systemMy->_systemMap->DataAssociation();
        }
    }

    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate 1", { 128.f, 32.f })) {
        if (_systemMy && _systemMy->_systemMap) {
            SystemMap& systemMap = *_systemMy->_systemMap;

            Body* star = systemMap.GetBody("Sun");
            if (star) {
                auto starPosT = star->GetPos();
                glm::vec3 starPos = glm::vec3(starPosT.x, starPosT.y, starPosT.z);
                float starMass = star->_mass;

                float dist = 1000.0f;
                int countX = 15;
                int countY = 15;

                for (int iX = -countX; iX < countX; ++iX) {
                    for (int iY = -countY; iY < countY; ++iY) {
                        if (iX == 0 && iY == 0) {
                            continue;
                        }

                        glm::vec3 pos(iX * dist, iY * dist, 0);
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
            SystemMap& systemMap = *_systemMy->_systemMap;

            Body* star = systemMap.GetBody("Sun");
            if (star) {
                auto starPosT = star->GetPos();
                glm::vec3 starPos = glm::vec3(starPosT.x, starPosT.y, starPosT.z);
                float starMass = star->_mass;

                size_t count = 999;
                float dDist = 25.f;
                float dist = 5000.f;
                float angle = 0.f;
                float dAngle = glm::pi<float>() / 10.f;

                for (size_t i = 0; i < count; ++i) {
                    float iX = dist * std::cos(angle) - dist * std::sin(angle);
                    float iY = dist * std::sin(angle) + dist * std::cos(angle);

                    glm::vec3 pos(iX, iY, 0);
                    float mass = 100.f;

                    glm::vec3 gravityVector = pos - starPos;
                    glm::vec3 normalizeGravityVector = glm::normalize(gravityVector);

                    float g90 = glm::pi<float>() / 2.0;
                    glm::vec3 velocity(normalizeGravityVector.x * std::cos(g90) - normalizeGravityVector.y * std::sin(g90),
                        normalizeGravityVector.x * std::sin(g90) + normalizeGravityVector.y * std::cos(g90),
                        0.f);

                    velocity *= std::sqrtf(systemMap._constGravity * starMass / glm::length(gravityVector));
                    systemMap.Add("BrownStone", pos, velocity, mass, "");

                    dist += dDist;
                    angle += dAngle;
                }
            }
        }
    }
}
