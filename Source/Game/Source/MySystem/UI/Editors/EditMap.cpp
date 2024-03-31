// ◦ Xyz ◦

#include "EditMap.h"
#include "Core.h"
#include "imgui.h"
#include "Common/Help.h"
#include "Object/Map.h"
#include "Object/Model.h"
#include "Draw/Camera/Camera.h"

namespace Editor {
    const std::string MapEditor::windowName = "Edit map";

    Object::Ptr MapEditor::NewObject() {
        if (MapEditor* windowPtr = dynamic_cast<MapEditor*>(UI::GetWindow(windowName).get())) {
            return windowPtr->_tempObjectPtr;
        }
        return nullptr;
    }

    void MapEditor::AddObjectToMap() {
        if (MapEditor* windowPtr = dynamic_cast<MapEditor*>(UI::GetWindow(windowName).get())) {
            Map::GetFirstCurrentMap().GetObjects().emplace_back(windowPtr->_tempObjectPtr);
            windowPtr->_tempObjectPtr = nullptr;
        }
    }

    //...
    MapEditor::MapEditor()
        : UI::Window(this)
    {
        _name[0] = '\0';
        _model[0] = '\0';
    }

    void MapEditor::OnOpen() {
        SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize);

        auto posWindow = ImGui::GetWindowPos();

        if (posWindow.x < 0.f) posWindow.x = 0.f;
        if (posWindow.y < 0.f) posWindow.y = 0.f;
        ImGui::SetWindowPos(Id().c_str(), posWindow);

        ImGui::SetWindowSize(Id().c_str(), { 750.f, 333.f });

        Load();
    }

    void MapEditor::OnClose() {
        _tempObjectPtr.reset();

        Save();
    }

    void MapEditor::Update() {
        if (_tempObjectPtr) {
            glm::vec3 cursorPos3;// NEW_CAMERA = Camera::current.corsorCoord();
            _tempObjectPtr->setPos(cursorPos3);
        }
    }

    //...
    void MapEditor::Draw() {
        _guiId = 0;

        ListDisplay();

        ImGui::SameLine();
        EditDisplay();

        ImGui::Dummy(ImVec2(0.f, 0.f));
        ButtonDisplay();

        CommonPopupModal::Draw();
    }

    //...
    void MapEditor::ListDisplay() {
        ImGui::BeginChild("list", { 210.f, 260.f }, true);

        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, 200.f);
        ImGui::SetColumnWidth(1, 1.f);

        for (Object::Ptr object : Map::GetFirstCurrentMap().GetObjects()) {
            bool select = _selectObjectPtr == object;
            ImGui::TextColored(select ? ImVec4(0.3f, 0.6f, 0.9f, 1.0f) : ImVec4(0.1f, 0.1f, 0.1f, 1.0f), "%s", object->getName().c_str());

            if (ImGui::IsItemHovered()) {
                if (ImGui::IsItemClicked()) {
                    _selectObjectPtr = object;
                    help::CopyToArrayChar(_name, object->getName());
                    help::CopyToArrayChar(_model, object->getModel().getName());
                }
            }
            ImGui::NextColumn();
            ImGui::NextColumn();
        }

        ImGui::EndChild();
    }

    void MapEditor::EditDisplay() {
        ImGui::BeginChild("edit", { 518.f, 260.f }, true);

        if (_selectObjectPtr) {
            ImGui::PushItemWidth(390.f);

            ImGui::SameLine(30.f);
            ImGui::Text("Name");
            ImGui::SameLine(65.f);
            ImGui::InputText("##name_input", _name.data(), _name.size());
            ImGui::Dummy(ImVec2(0.f, 0.f));

            //...
            ImGui::SameLine(24.f);
            ImGui::Text("Object");
            ImGui::SameLine(65.f);
            ImGui::InputText("##object_input", _model.data(), _model.size());
            ImGui::SameLine();
            if (ImGui::Button("?##choose_object_btn", { 20.f, 20.f })) {
                //...
            }
            ImGui::Dummy(ImVec2(0.f, 0.f));
        }

        ImGui::EndChild();
    }

    void MapEditor::ButtonDisplay() {
        ImGui::BeginChild("button", { 750.f, 30.f }, false);

        //...
        if (ImGui::Button("Add", { 62.f, 24.f })) {
            Editor::stringPtr nameModelPtr(new std::string());

            CommonPopupModal::Show(GetPtr(), [this, nameModelPtr]() {
                for (const auto& modelPair : Model::GetMap()) {
                    bool select = modelPair.first == *nameModelPtr;
                    ImGui::TextColored(select ? ImVec4(0.3f, 0.6f, 0.9f, 1.0f) : ImVec4(0.1f, 0.1f, 0.1f, 1.0f), "%s", modelPair.first.c_str());

                    if (ImGui::IsItemHovered()) {
                        if (ImGui::IsItemClicked()) {
                            *nameModelPtr = modelPair.first;

                            _tempObjectPtr.reset();
                            glm::vec3 cursorPos3; // NEW_CAMERA = Camera::current.corsorCoord();
                            _tempObjectPtr = std::make_shared<Object>(("##NEW:"+ modelPair.first), modelPair.first, cursorPos3);
                        }
                    }
                }

                ImGui::Dummy(ImVec2(0.f, 0.f));
                if (ImGui::Button("Close", { 100.f, 24.f })) {
                    _tempObjectPtr.reset();
                    CommonPopupModal::Hide();
                }
                }, "Add");
        }

        ImGui::SameLine();
        if (ImGui::Button("Copy", { 62.f, 24.f }) && _selectObjectPtr) {
            //...
        }

        ImGui::SameLine();
        if (ImGui::Button("Remove", { 62.f, 24.f })) {
            //...
        }

        //...
        ImGui::SameLine(220.f);
        if (ImGui::Button("Apply", { 62.f, 24.f })) {
            //Apply();
        }

        ImGui::SameLine();
        if (ImGui::Button("Reset", { 62.f, 24.f })) {
            //...
        }

        //...
        ImGui::SameLine(600.f);
        if (ImGui::Button("Save", { 62.f, 24.f })) {
            Save();
        }

        ImGui::SameLine();
        if (ImGui::Button("Close", { 62.f, 24.f })) {
            Close();
        }

        ImGui::EndChild();
    }

    //...
    void MapEditor::Load() {
    }

    void MapEditor::Save() {
    }
}