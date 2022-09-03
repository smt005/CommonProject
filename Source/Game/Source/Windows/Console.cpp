
#include "Console.h"
#include "Core.h"
#include "Screen.h"
#include "Callback/Callback.h"
#include "imgui.h"

namespace Editor {
    bool Console::lock = false;

    Console::Console() {
        SetId("Console");
        SetTitle("Console");
        VisibleCloseBtn(false);
        SetFlag(ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground);
        _buf = new char[128];
        _buf[0] = '\0';
        lock = true;
    }

    Console::~Console() {
        delete[] _buf;
        lock = false;
    }

    void Console::Draw() {
        _guiId = 0;
        ImGui::SetWindowPos(Id().c_str(), { 0.f, 0.f });

        ImGui::PushItemWidth(Engine::Screen::width() - 150.f);
        ImGui::PushID(++_guiId);
        ImGui::InputText("", _buf, 128);
        ImGui::PopID();
        ImGui::PopItemWidth();

        ImGui::SameLine();
        ImGui::PushID(++_guiId);
        if (ImGui::Button("Apply", ImVec2(40.f, 20.f))) {
            std::string text(_buf);
            if (text == "close") {
                //Engine::Core::close();
            }
        }
        ImGui::PopID();

        ImGui::SameLine();
        ImGui::PushID(++_guiId);
        if (ImGui::Button(":", ImVec2(20.f, 20.f))) {
            _showDemo = !_showDemo;
        }
        ImGui::PopID();

        ImGui::SameLine();
        ImGui::PushID(++_guiId);
        if (ImGui::Button("?", ImVec2(20.f, 20.f))) {
            _showDemo = !_showDemo;
        }
        ImGui::PopID();

        ImGui::SameLine();
        ImGui::PushID(++_guiId);
        if (ImGui::Button("x", ImVec2(20.f, 20.f))) {
            UI::CloseWindow(this);
        }
        ImGui::PopID();

        //...
        Commands();

        //...
        Demo();
    }

    void Console::Demo() {
        if (!_showDemo) {
            return;
        }

        static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

        if (_show_demo_window)
            ImGui::ShowDemoWindow(&_show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            static float f = 0.0f;

            ImGui::Begin("Hello, world!");                                // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");                     // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &_show_demo_window);            // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &_show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);                  // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color);       // Edit 3 floats representing a color

            if (ImGui::Button("Button"))                                  // Buttons return true when clicked (most widgets return true when edited/activated) 
            {
                _counter++;
                if (_counter > 5) {
                    Close();
                }
            }

            ImGui::SameLine();
            ImGui::Text("counter = %d", _counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // 3. Show another simple window.
        if (_show_another_window)
        {
            ImGui::Begin("Another Window", &_show_another_window);         // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                _show_another_window = false;
            ImGui::End();
        }
    }

    void Console::Commands() {
        if (ImGui::IsKeyReleased(525)) { // Enter
            std::string text(_buf);
            if (text == "close") {
                Engine::Core::close();
            }
        }
    }
}
