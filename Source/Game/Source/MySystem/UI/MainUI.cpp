// ◦ Xyz ◦
#include "MainUI.h"
#include <imgui.h>
#include "TopUI.h"
#include "BottomUI.h"
#include "SpaceManagerUI.h"
#include "ListHeaviestUI.h"
#include "ComputationsUI.h"
#include "CommonData.h"
#include "Editors/CommandsWindow.h"
#include "Editors/QuestsWindow.h"
#include "Editors/QuestsEditorWindow.h"
#include "Editors/EditorModel.h"
#include "Editors/EditMap.h"
#include <Draw/Camera/Camera.h>
#include <Draw/Camera/CameraControlOutside.h>
#include <Callback/Callback.h>
#include <Callback/CallbackEvent.h>
#include <Draw/DrawLight.h>
#include <MySystem/MySystem.h>
#include "../Objects/SpaceManager.h"
#include "../Objects/BodyData.h"
#include "../Objects/Space.h"
#include <Object/Object.h>
#include <Object/Model.h>

#include <Draw2/Draw2.h>
#include <Draw2/Shader/ShaderDefault.h>

#include "../Commands/Functions.h"
#include <MyStl/Event.h>
#include "../Commands/Events.h"

namespace
{
	Engine::Callback callback;
	Object::Ptr bodyMarker;
	Model::Ptr cursorModel;
}

void MainUI::Open()
{
	InitCallback();

	UI::ShowWindow<TopUI>();
	UI::ShowWindow<BottomUI>();
}

void MainUI::Hide()
{
	UI::CloseWindowT<SpaceManagerUI>();
	UI::CloseWindowT<ListHeaviestUI>();
}

void MainUI::InitCallback()
{
	callback.add(Engine::CallbackType::PRESS_KEY, [](const Engine::CallbackEventPtr& callbackEventPtr) {
		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (key == Engine::VirtualKey::F1) {
			if (UI::ShowingWindow<ComputationsUI>()) {
				UI::CloseWindowT<ComputationsUI>();
			}
			else {
				UI::ShowWindow<ComputationsUI>();
			}
		}

		if (key == Engine::VirtualKey::F2) {
			if (UI::ShowingWindow<SpaceManagerUI>()) {
				UI::CloseWindowT<SpaceManagerUI>();
			}
			else {
				UI::ShowWindow<SpaceManagerUI>();
			}
		}

		if (key == Engine::VirtualKey::F3) {
			if (UI::ShowingWindow<ListHeaviestUI>()) {
				UI::CloseWindowT<ListHeaviestUI>();
			}
			else {
				UI::ShowWindow<ListHeaviestUI>();
			}
		}

		if (key == Engine::VirtualKey::F4) {
			if (UI::ShowingWindow<CommandsWindow>()) {
				UI::CloseWindowT<CommandsWindow>();
			}
			else {
				UI::ShowWindow<CommandsWindow>();
			}
		}

		if (key == Engine::VirtualKey::F5) {
			if (UI::ShowingWindow<QuestsWindow>()) {
				UI::CloseWindowT<QuestsWindow>();
			}
			else {
				UI::ShowWindow<QuestsWindow>();
			}
		}

		// Editor
		if (key == Engine::VirtualKey::F9) {
			if (UI::ShowingWindow<Editor::ModelEditor>()) {
				UI::CloseWindowT<Editor::ModelEditor>();
			}
			else {
				UI::ShowWindow<Editor::ModelEditor>();
			}
		}

		if (key == Engine::VirtualKey::F10) {
			if (UI::ShowingWindow<Editor::MapEditor>()) {
				UI::CloseWindowT<Editor::MapEditor>();
			}
			else {
				UI::ShowWindow<Editor::MapEditor>();
			}
		}

		if (key == Engine::VirtualKey::F11) {
			if (UI::ShowingWindow<Editor::QuestsEditorWindow>()) {
				UI::CloseWindowT<Editor::QuestsEditorWindow>();
			}
			else {
				UI::ShowWindow<Editor::QuestsEditorWindow>();
			}
		}

		if (key == Engine::VirtualKey::F12) {
			//...
		}

		// Цифровые кнопки
		if (key == Engine::VirtualKey::VK_0) {
			CommonData::bool0 = !CommonData::bool0;
		}
		else if (key == Engine::VirtualKey::VK_1) {
			CommonData::bool1 = !CommonData::bool1;
		}
		else if (key == Engine::VirtualKey::VK_2) {
			CommonData::bool2 = !CommonData::bool2;
		}
		else if (key == Engine::VirtualKey::VK_3) {
			CommonData::bool3 = !CommonData::bool3;
		}
		else if (key == Engine::VirtualKey::VK_4) {
			CommonData::bool4 = !CommonData::bool4;
		}
		else if (key == Engine::VirtualKey::VK_5) {
			CommonData::bool5 = !CommonData::bool5;
		}
		else if (key == Engine::VirtualKey::VK_6) {
			CommonData::bool6 = !CommonData::bool6;
		}
		else if (key == Engine::VirtualKey::VK_7) {
			CommonData::bool7 = !CommonData::bool7;
		}
		else if (key == Engine::VirtualKey::VK_8) {
			CommonData::bool8 = !CommonData::bool8;
		}
		else if (key == Engine::VirtualKey::VK_9) {
			CommonData::bool9 = !CommonData::bool9;
		}
	});

	callback.add(Engine::CallbackType::PRESS_TAP, [](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (CommonData::IsLockScreen() > 0) {
			return;
		}
		if (IsLockAction()) {
			return;
		}

		if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			if (tapCallbackEvent->_id == Engine::VirtualTap::LEFT) {
				EventOnTap::Instance().Action();
			}
		}
	});

	callback.add(Engine::CallbackType::RELEASE_TAP, [](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (IsLockAction()) {
			return;
		}

		/*if (MySystem::currentSpace;  Engine::TapCallbackEvent * tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			if (tapCallbackEvent->_id == Engine::VirtualTap::MIDDLE) {
				const glm::mat4x4& matCamera = _mySystem->_camearCurrent->ProjectView();
				MySystem::currentSpace->_focusBody = MySystem::currentSpace->HitObject(matCamera);
				MySystem::currentSpace->_selectBody = MySystem::currentSpace->_focusBody;
			}

			if (tapCallbackEvent->_id == Engine::VirtualTap::LEFT) {
				if (BottomUI* bottomUI = dynamic_cast<BottomUI*>(UI::GetWindow<BottomUI>().get())) {
					if (bottomUI->_addBodyType == AddBodyType::ORBIT) {
						auto cursorPosGlm = _mySystem->_camearCurrent->corsorCoord();
						Math::Vector3 cursorPos(cursorPosGlm.x, cursorPosGlm.y, cursorPosGlm.z);
						SpaceManager::AddObjectOnOrbit(MySystem::currentSpace.get(), cursorPos);
						bottomUI->_addBodyType = AddBodyType::NONE;
					}
				}
			}
		}*/
	});
}

void MainUI::DrawOnSpace()
{
	/*if (!bodyMarker) {
		bodyMarker = std::make_shared<Object>("Marker", "Marker");
	}

	const glm::mat4x4& matCamera = _mySystem->_camearCurrent->ProjectView();

	Camera::Set(_mySystem->_camearScreen);
	ShaderDefault::current->Use();

	for (Body::Ptr& body : MySystem::currentSpace->_bodies) {	
		Math::Vector3 posOnScreen = body->PosOnScreen(matCamera, false);
		float pos[] = { posOnScreen.x, posOnScreen.y, posOnScreen.z };
		
		bodyMarker->setPos(pos);

		Draw2::SetModelMatrix(bodyMarker->getMatrix());
		Draw2::Draw(bodyMarker->getModel());
	}*/

	// Курсор
	if (cursorModel && MySystem::_camearSide) {
		Camera::Set<Camera>(MySystem::_camearSide);
		Draw2::DepthTest(false);
		ShaderDefault::Instance().Use();

		float color4[] = { 1.f, 1.f, 1.f, 1.f };
		Draw2::SetColorClass<ShaderDefault>(color4);

		auto mousePos = MySystem::_camearSide->corsorCoord();
		glm::mat4x4 mouseMat(1.f);
		mouseMat[3][0] = mousePos.x;
		mouseMat[3][1] = mousePos.y;
		mouseMat[3][2] = mousePos.z;
		Draw2::SetModelMatrixClass<ShaderDefault>(mouseMat);

		Draw2::Draw(*cursorModel);
	}
}

bool MainUI::IsLockAction()
{
	return ImGui::GetIO().WantCaptureMouse;
}

unsigned int MainUI::GetViewType()
{
	if (BottomUI* bottomUI = dynamic_cast<BottomUI*>(UI::GetWindow<BottomUI>().get())) {
		return (unsigned int)bottomUI->_viewType;
	}
}

void MainUI::SetCursorModel(const std::string& nameModel)
{
	if (!nameModel.empty()) {
		cursorModel = Model::getByName(nameModel);
	}
	else {
		cursorModel.reset();
	}
}
