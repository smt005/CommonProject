// ◦ Xyz ◦

#include "EditorModel.h"
#include "Core.h"
#include "imgui.h"

#include "Object/Model.h"
#include "FileManager.h"
#include "Common/Help.h"
#include "Common/WindowsWin32.h"

namespace Editor {

    ModelEditor::ModelEditor()
        : UI::Window(this)
        , _selectNamePtr(nullptr)
    {
        _name[0] = '\0';
        _shape[0] = '\0';
        _texture[0] = '\0';
        _physicShape[0] = '\0';
        _scale[0] = '\0';
    }

    void ModelEditor::OnOpen() {
        SetFlag(ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize);

        auto posWindow = ImGui::GetWindowPos();

        if (posWindow.x < 0.f) posWindow.x = 0.f;
        if (posWindow.y < 0.f) posWindow.y = 0.f;
        ImGui::SetWindowPos(Id().c_str(), posWindow);

        ImGui::SetWindowSize(Id().c_str(), { 750.f, 333.f });

        Load();
    }

    void ModelEditor::OnClose() {
        Save();
    }

    void ModelEditor::Draw() {
        _guiId = 0;

        ListDisplay();

        ImGui::SameLine();
        EditDisplay();

        ImGui::Dummy(ImVec2(0.f, 0.f));
        ButtonDisplay();

        CommonPopupModal::Draw();
    }

    //...
    void ModelEditor::ListDisplay() {
        ImGui::BeginChild("list", { 210.f, 260.f }, true);

        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, 200.f);
        ImGui::SetColumnWidth(1, 1.f);

        for (std::string& name : _modelNames) {
            bool select = _selectNamePtr  && (*_selectNamePtr == name);
            ImGui::TextColored(select ? ImVec4(0.3f, 0.6f, 0.9f, 1.0f) : ImVec4(0.9f, 0.9f, 0.9f, 1.0f), "%s", name.c_str());
            if (ImGui::IsItemHovered()) {
                if (ImGui::IsItemClicked()) {
                    Select(name);
                }
            }
            ImGui::NextColumn();
            ImGui::NextColumn();
        }

        ImGui::EndChild();
    }

    void ModelEditor::EditDisplay() {
        ImGui::BeginChild("edit", { 518.f, 260.f }, true);

        if (_selectNamePtr && !_selectNamePtr->empty()) {
            ImGui::PushItemWidth(390.f);

            ImGui::SameLine(30.f);
            ImGui::Text("Name");
            ImGui::SameLine(65.f);
            ImGui::InputText("##name_input", _name.data(), _name.size());
            ImGui::Dummy(ImVec2(0.f, 0.f));

            //...
            ImGui::SameLine(24.f);
            ImGui::Text("Shape");
            ImGui::SameLine(65.f);
            ImGui::InputText("##shape_input", _shape.data(), _shape.size());
            ImGui::SameLine();
            if (ImGui::Button("?##choose_shape_btn", {20.f, 20.f})) {
                ChooseFile(_shape);
            }
            ImGui::SameLine();
            if (ImGui::Button(">##exp_shape_btn", { 20.f, 20.f })) {
                OpenExplorer(_shape.data());
            }
            ImGui::Dummy(ImVec2(0.f, 0.f));

            //...
            ImGui::SameLine(10.f);
            ImGui::Text("Texture");
            ImGui::SameLine(65.f);
            ImGui::InputText("##texture_input", _texture.data(), _texture.size());
            ImGui::SameLine();
            if (ImGui::Button("?##choose_texture_btn", { 20.f, 20.f })) {
                ChooseFile(_texture);
            }
            ImGui::SameLine();
            if (ImGui::Button(">##exp_texture_btn", { 20.f, 20.f })) {
                OpenExplorer(_texture.data());
            }
            ImGui::Dummy(ImVec2(0.f, 0.f));

            //...
            ImGui::SameLine(14.f);
            ImGui::Text("Physic");
            ImGui::SameLine(65.f);
            ImGui::InputText("##physic_input", _physicShape.data(), _physicShape.size());
            ImGui::SameLine();
            if (ImGui::Button("?##choose_physic_btn", { 20.f, 20.f })) {
                ChooseFile(_physicShape);
            }
            ImGui::SameLine();
            if (ImGui::Button(">##exp_physic_btn", { 20.f, 20.f })) {
                OpenExplorer(_physicShape.data());
            }
            ImGui::Dummy(ImVec2(0.f, 0.f));

            //...
            ImGui::SameLine(24.f);
            ImGui::Text("Scale");
            ImGui::SameLine(65.f);
            ImGui::InputText("##scale_input", _scale.data(), _scale.size());
            ImGui::Dummy(ImVec2(0.f, 0.f));

            //...
            if (ModelPtr model = Model::getByName(_name.data())) {                
                ImGui::Image(reinterpret_cast<ImTextureID>(model->textureId()), { 100.f, 100.f }, {0, 1}, {1, 0}, { 1.f, 1.f, 1.f, 1.f }, { 1.f, 1.f, 1.f, 1.f });
            }
        }

        ImGui::EndChild();
    }

    void ModelEditor::ButtonDisplay() {
        ImGui::BeginChild("button", { 750.f, 30.f }, false);

        //...
        if (ImGui::Button("Add", { 62.f, 24.f })) {
            Editor::TextCharPtr newNamePtr(new Editor::TextChar());
            (*newNamePtr)[0] = '\0';

            CommonPopupModal::Show(GetPtr(), [this, newNamePtr]() {
                ImGui::InputText("##new_name_input", newNamePtr->data(), newNamePtr->size());

                ImGui::Dummy(ImVec2(0.f, 0.f));
                if (ImGui::Button("Add", { 100.f, 24.f }) && (*newNamePtr)[0] != '\0') {
                    Add(std::string(newNamePtr->data()));
                    CommonPopupModal::Hide();
                }

                ImGui::SameLine();
                if (ImGui::Button("Close", { 100.f, 24.f })) {
                    CommonPopupModal::Hide();
                }
                }, "Add");
        }

        ImGui::SameLine();
        if (ImGui::Button("Copy", { 62.f, 24.f }) && _selectNamePtr) {
            Editor::TextCharPtr newNamePtr(new Editor::TextChar());
            (*newNamePtr)[0] = '\0';

            CommonPopupModal::Show(GetPtr(), [this, newNamePtr]() {
                ImGui::Text("Copy %s", _selectNamePtr->c_str());

                ImGui::Dummy(ImVec2(0.f, 0.f));
                ImGui::InputText("##new_name_input", newNamePtr->data(), newNamePtr->size());

                ImGui::Dummy(ImVec2(0.f, 0.f));
                if (ImGui::Button("Add", { 100.f, 24.f }) && (*newNamePtr)[0] != '\0') {
                    Copy(std::string(newNamePtr->data()), *_selectNamePtr);
                    CommonPopupModal::Hide();
                }

                ImGui::SameLine();
                ImGui::SameLine();
                if (ImGui::Button("Close", { 100.f, 24.f })) {
                    CommonPopupModal::Hide();
                }
                }, "Add");
        }

        ImGui::SameLine();
        if (ImGui::Button("Remove", { 62.f, 24.f })) {
            CommonPopupModal::Show(GetPtr(), [this]() {
                ImGui::TextColored({ 0.9f, 0.0f, 0.0f, 1.0f }, "Remove %s?", _selectNamePtr ? _selectNamePtr->c_str() : "ERROR");

                ImGui::Dummy(ImVec2(0.f, 0.f));
                if (ImGui::Button("Remove", { 100.f, 24.f })) {
                    Remove(*_selectNamePtr);
                    CommonPopupModal::Hide();
                }

                ImGui::SameLine();
                ImGui::SameLine();
                if (ImGui::Button("Close", { 100.f, 24.f })) {
                    CommonPopupModal::Hide();
                }
                }, "Remove");
        }

        //...
        ImGui::SameLine(220.f);
        if (ImGui::Button("Apply", { 62.f, 24.f})) {
            Apply();
        }

        ImGui::SameLine();
        if (ImGui::Button("Reset", { 62.f, 24.f })) {
            CommonPopupModal::Show(GetPtr(), [this]() {
                ImGui::TextColored({ 0.9f, 0.0f, 0.0f, 1.0f }, "Reset %s?", _selectNamePtr ? _selectNamePtr->c_str() : "ERROR");


                ImGui::Dummy(ImVec2(0.f, 0.f));
                if (ImGui::Button("Reset", { 100.f, 24.f })) {
                    Select(*_selectNamePtr);
                    CommonPopupModal::Hide();
                }

                ImGui::SameLine();
                ImGui::SameLine();
                if (ImGui::Button("Close", { 100.f, 24.f })) {
                    CommonPopupModal::Hide();
                }
            }, "Reset");
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
    void ModelEditor::Remove(const std::string& name) {
        if (_selectNamePtr && !_selectNamePtr->empty()) {
            std::string newName = _name.data();
            Json::Value& data = Model::_data;

            data.removeMember(*_selectNamePtr);

            auto itErase = std::find_if(_modelNames.begin(), _modelNames.end(), [this](const std::string name) { return *_selectNamePtr == name; });
            _modelNames.erase(itErase);
            _selectNamePtr = nullptr;
        }
    }

    void ModelEditor::Add(const std::string& newName) {
        if (std::find_if(_modelNames.begin(), _modelNames.end(), [&newName](const std::string& name) { return name == newName; }) != _modelNames.end()) {
            return;
        }

        _selectNamePtr = &_modelNames.emplace_back(newName);

        Json::Value& data = Model::_data;
        data[newName]["shape"] = "";
        data[newName]["texture"] = "";

        help::CopyToArrayChar(_name, newName);
        _shape[0] = '\0';
        _texture[0] = '\0';
        //_scale[0] = '\0';
    }

    void ModelEditor::Copy(std::string newName, const std::string copyName) {
        if (std::find_if(_modelNames.begin(), _modelNames.end(), [&newName](const std::string& name) { return name == newName; }) != _modelNames.end()) {
            return;
        }

        _selectNamePtr = &_modelNames.emplace_back(newName);

        Json::Value& data = Model::_data;

        const std::string shape = data[copyName]["shape"].asString();
        if (!shape.empty()) {
            data[newName]["shape"] = shape;
        }

        const std::string texture = data[copyName]["texture"].asString();
        if (!texture.empty()) {
            data[newName]["texture"] = texture;
        }

        help::CopyToArrayChar(_name, newName);
        help::CopyToArrayChar(_shape, shape);
        help::CopyToArrayChar(_texture, texture);
        //_scale[0] = '\0';
    }

    void ModelEditor::Select(std::string& name) {
        Json::Value& data = Model::_data;
        Json::Value& dataModel = data[name];
        if (dataModel.empty()) {
            data[name]["shape"] = "";
            data[name]["texture"] = "";
        }

        const std::string shape = dataModel["shape"].empty() ? "" : dataModel["shape"].asString();
        const std::string texture = dataModel["texture"].empty() ? "" : dataModel["texture"].asString();
        //const std::string scale = dataModel["scale"].empty() ? "" : dataModel["scale"].asString();

        help::CopyToArrayChar(_name, name);
        help::CopyToArrayChar(_shape, shape);
        help::CopyToArrayChar(_texture, texture);
        help::CopyToArrayChar(_physicShape, ""s);
        help::CopyToArrayChar(_scale, ""s);

        /*if (scale.empty()) {
            _scale[0] = '1';
            _scale[0] = '\0';
        } else {
            help::CopyToArrayChar(_scale, scale);
        }*/

        _selectNamePtr = &name;
    }

    void ModelEditor::Apply() {
        if (_selectNamePtr && !_selectNamePtr->empty()) {
            std::string newName = _name.data();
            Json::Value& data = Model::_data;

            data.removeMember(*_selectNamePtr);

            data[newName]["shape"] = std::string(_shape.data());
            data[newName]["texture"] = std::string(_texture.data());

            /*if (_scale[0] != '\0') {
                data[newName]["scale"] = std::string(_scale.data());
            } else {
                data[newName]["scale"] = '1';
            }*/

            *_selectNamePtr = newName;

            Model::getByName(newName)->load();
        }
    }

    void ModelEditor::ChooseFile(Editor::TextChar& textChar) {
        std::string fiilNameFile = Engine::SelectFile();
        std::replace(fiilNameFile.begin(), fiilNameFile.end(), '\\', '/');

        std::string resourcesDir = Engine::FileManager::getResourcesDir().string();
        std::replace(resourcesDir.begin(), resourcesDir.end(), '\\', '/');

        size_t pos = fiilNameFile.find(resourcesDir);
        if (pos == resourcesDir.npos) {
            return;
        }

        pos += resourcesDir.size() + 1;
        if (pos >= fiilNameFile.size()) {
            return;
        }

        size_t size = fiilNameFile.size() - pos;
        std::string nameFile = fiilNameFile.substr(pos, size);

        help::CopyToArrayChar(textChar, nameFile);
    }

    void ModelEditor::OpenExplorer(const std::string& name) {
        if (!name.empty()) {
            Engine::OpenFolderAndSelectItems(name, Engine::FileManager::getResourcesDir().string());
        }
    }

    //...
    void ModelEditor::Load() {
        Json::Value& data = Model::_data;

        for (auto it = data.begin(); it != data.end(); ++it) {
            if (!it->isObject() ) {
                continue;
            }

            const string& nameModel = it.name();
            if (nameModel == "default") {
                continue;
            }

            _modelNames.emplace_back(nameModel);
        }
    }

    void ModelEditor::Save() {
        help::saveJson(Model::modelPath, Model::_data);
    }
}
