
#include "SystemManager.h"
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

            Body* star = &systemMap.RefFocusBody();
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


    ImGui::Dummy(ImVec2(0.f, 0.f));
    if (ImGui::Button("Generate 3", { 128.f, 32.f })) {
        if (_systemMy && _systemMy->_systemMap) {
            SystemMap& systemMap = *_systemMy->_systemMap;


            Body* star = &systemMap.RefFocusBody();
            if (star) {
                auto starPosT = star->GetPos();
                glm::vec3 starPos = glm::vec3(starPosT.x, starPosT.y, starPosT.z);
                double starMass = star->_mass;

                unsigned int countLevel = 3;
                double dR = 1000;
                double offA = 0;
                double dA = glm::pi<double>() / 6;
                double halfA = glm::pi<double>() / 12;

                struct Point {
                    double x, y, z;
                    Point() : x(0), y(0), z(0) {}
                    Point(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
                    bool hit(const Point& point, double error = 10) {
                        //return (std::abs(x - point.x) < error && std::abs(y - point.y) < error);

                        if (std::abs(x - point.x) < error && std::abs(y - point.y) < error) {
                            return true;
                        } else {
                            return false;
                        }
                    }
                };

                Point lastPoint(starPosT.x, starPosT.y, starPosT.z);
                double radius = 0;
                double offsetAngle = 0;
                double angle = 0;

                bool needSavePoint = false;

                while (countLevel > 0) {
                    Point point;
                    float a = offsetAngle + angle;
                    point.x = radius * std::cos(a) - radius * std::sin(a);
                    point.y = radius * std::sin(a) + radius * std::cos(a);
                    point.z = 0;

                    if ((int)point.x == 707 || (int)point.y == 1224 || (int)point.y == 1225) {
                        std::cout << "..." << std::endl;
                    }

                    if (needSavePoint) {
                        lastPoint = point;
                        needSavePoint = false;
                    } else {
                        if (!lastPoint.hit(point)) {
                            std::cout << "[" << point.x << ' ' << point.y << ' ' << point.z << "] a: " << a  << " off: " << offsetAngle << " AAA: " << angle  << " r: " << radius << " CREATE" << std::endl;
                            CreateOrbitBody(point.x, point.y, point.z);
                            lastPoint = point;
                            angle += dA;
                        } else {
                            radius += dR;
                            offsetAngle = angle + halfA;
                            needSavePoint = true;
                            --countLevel;
                            angle = 0;
                        }
                    }
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
