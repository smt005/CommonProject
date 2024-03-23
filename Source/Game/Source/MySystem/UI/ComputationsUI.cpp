// ◦ Xyz ◦

#include "ComputationsUI.h"
#include "MySystem/MySystem.h"
#include "../Objects/SpaceManager.h"

#include <cmath>
#include "imgui.h"
#include <glm/ext/scalar_constants.hpp>
#include "CommonData.h"
#include "Math/Vector.h"
#include "../Objects/BodyData.h"
#include "../Objects/Space.h"
#include "../Objects/SpaceTree02.h"
#include <../../CUDA/Source/Wrapper.h>
#include <../../CUDA/Source/Emulate.h>

namespace {
    Space::Ptr spaceTemp;
}

ComputationsUI::ComputationsUI() : UI::Window(this) {
}

void ComputationsUI::OnOpen() {
    SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize);

    ImGui::SetWindowPos(Id().c_str(), { _x, _y });
    ImGui::SetWindowSize(Id().c_str(), { _width, _height });

    ++CommonData::lockAction;
}

void ComputationsUI::OnClose() {
    --CommonData::lockAction;
}

void ComputationsUI::Update() {

}

void ComputationsUI::Draw() {
    if (!MySystem::currentSpace) {
        return;
    }

    Space* currentSpacePtr = MySystem::currentSpace.get();
    std::string currentClass = currentSpacePtr->GetNameClass();
    ImGui::Text(currentClass.c_str());

    ImGui::Separator();
    {
        std::string btnText = "PROCESS: ";
        btnText += currentSpacePtr->processGPU ? "GPU" : "CPU";
        ImGui::PushStyleColor(ImGuiCol_Button, currentSpacePtr->processGPU ? Editor::blueColor : Editor::defaultColor);

        if (ImGui::Button(btnText.c_str(), { 128.f, 32.f })) {
            currentSpacePtr->processGPU = !currentSpacePtr->processGPU;
            CUDA::processGPU = currentSpacePtr->processGPU;
        }
        ImGui::PopStyleColor();
    }
    {
        ImGui::Dummy(ImVec2(0.f, 0.f));
        std::string btnText = "THREAD: ";
        btnText += currentSpacePtr->multithread ? "ON" : "OFF";
        ImGui::PushStyleColor(ImGuiCol_Button, currentSpacePtr->multithread ? Editor::blueColor : Editor::defaultColor);

        if (ImGui::Button(btnText.c_str(), { 128.f, 32.f })) {
            currentSpacePtr->multithread = !currentSpacePtr->multithread;
            CUDA::multithread = currentSpacePtr->multithread;
        }
        ImGui::PopStyleColor();
    }
    {
        ImGui::Dummy(ImVec2(0.f, 0.f));
        static int tagInt = 0;
        if (ImGui::InputInt("tag: ", &tagInt)) {
            currentSpacePtr->tag = tagInt;
            currentSpacePtr->Preparation();
        }   
    }

    //................................................................
    ImGui::Separator();
    ImGui::BeginChild("Classes", { 140.f, 100.f }, false);

    for (const std::string& className : SpaceManager::GetListClasses()) {
        ImGui::PushStyleColor(ImGuiCol_Button, currentClass == className ? Editor::greenColor : Editor::defaultColor);
    
        if (ImGui::Button(className.c_str(), { 128.f, 32.f }) && currentClass != className) {
            std::shared_ptr<Space> space = SpaceManager::CopySpace(className, currentSpacePtr);
            spaceTemp = MySystem::currentSpace;
            space->Preparation();
            MySystem::currentSpace = space;
        }

        ImGui::PopStyleColor();
    }

    ImGui::EndChild();

    //................................................................
    ImGui::Separator();
    ImGui::BeginChild("Params", { 140.f, 100.f }, false);

    if (SpaceTree02* spacePtr = dynamic_cast<SpaceTree02*>(currentSpacePtr)) {
        if (ImGui::Button("Reset CO", {128.f, 32.f})) {
            spacePtr->searchOptimalCountBodies = true;
        }
    }

    ImGui::EndChild();

    ImGui::Separator();
}
