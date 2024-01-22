
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
#include <../../CUDA/Emulate.h>

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

                double minSpaceRange = 10000;
                double spaceRange = 50000;
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
                std::string countStr = _mySystem->_space->_params["COUNT"];
                if (!countStr.empty()) {
                    count = std::stoi(countStr);
                }

                _mySystem->_space->_bodies.reserve(count);

                double minSpaceRange = 10000;
                double spaceRange = 50000;
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
    {
            std::string btnText = "PROCESS: ";
            btnText += _mySystem->_space->processGPU ? "GPU" : "CPU";
            if (ImGui::Button(btnText.c_str(), { 128.f, 32.f })) {
                _mySystem->_space->processGPU = !_mySystem->_space->processGPU;
            }

            ImGui::Dummy(ImVec2(0.f, 0.f));

            static int tagInt = 0;
            if (ImGui::InputInt("tag: ", &tagInt)) {
                _mySystem->_space->tag = tagInt;
            }
    }

    ImGui::Separator();
    {
        ImGui::Dummy(ImVec2(0.f, 0.f));

        if (ImGui::Button("Test CUDA##test_cuda", { 128.f, 32.f })) {
            CUDA_TEST::Test(100);

            CUDA_TEST::Test(100000);
            CUDA_TEST::Test(50000, true);
            CUDA_TEST::Test(50000, false);
            CUDA_TEST::Test(10000, true);
            CUDA_TEST::Test(10000, false);
            CUDA_TEST::Test(1024);
            CUDA_TEST::Test(1000);
            CUDA_TEST::Test(100);
        }
    }

    ImGui::Separator();
    ImGui::BeginChild("Classes", { 140.f, 100.f }, false);

    for (const std::string& className : SpaceManager::GetListClasses()) {
        ImGui::PushStyleColor(ImGuiCol_Button, currentClass == className ? Editor::greenColor : Editor::defaultColor);
    
        if (ImGui::Button(className.c_str(), { 128.f, 32.f }) && currentClass != className) {
            std::shared_ptr<Space> space = SpaceManager::CopySpace(className, currentSpacePtr);
            spaceTemp = _mySystem->_space;
            space->Preparation();
            _mySystem->_space = space;
        }

        ImGui::PopStyleColor();
    }

    ImGui::EndChild();

    ImGui::Separator();
}
