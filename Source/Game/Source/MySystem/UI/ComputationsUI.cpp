
#include "ComputationsUI.h"
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
#include <../../CUDA/Source/Wrapper.h>
#include <../../CUDA/Source/Emulate.h>

namespace {
    Space::Ptr spaceTemp;
}

ComputationsUI::ComputationsUI() : UI::Window(this) {
    Close();
}

ComputationsUI::ComputationsUI(MySystem* mySystem)
    : UI::Window(this)
    , _mySystem(mySystem)
{}

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
    if (!_mySystem || !_mySystem->_space) {
        return;
    }

    Space* currentSpacePtr = _mySystem->_space.get();
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

    /*{
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
    }*/
}
