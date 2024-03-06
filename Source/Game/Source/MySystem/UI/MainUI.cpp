#include "MainUI.h"
#include "imgui.h"
#include "TopUI.h"
#include "BottomUI.h"
#include "SpaceManagerUI.h"
#include "ListHeaviestUI.h"
#include "ComputationsUI.h"
#include "Debug/CommandsWindow.h""
#include "Draw/Camera/CameraControlOutside.h"
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"
#include "Draw/DrawLight.h"
#include "MySystem/MySystem.h"
#include "../Objects/SpaceManager.h"
#include "../Objects/BodyData.h"
#include "../Objects/Space.h"
#include "Object/Object.h"
#include "Object/Model.h"

#include <Draw2/Draw2.h>
#include <Draw2/Shader/ShaderDefault.h>

namespace {
	Engine::Callback callback;
	Object::Ptr bodyMarker;
}

MySystem* MainUI::_mySystem = nullptr;

void MainUI::Open(MySystem* mySystem) {
	_mySystem = mySystem;

	InitCallback();

	UI::ShowWindow<TopUI>(_mySystem);
	UI::ShowWindow<BottomUI>(_mySystem);
}

void MainUI::Hide() {
	UI::CloseWindowT<SpaceManagerUI>();
	UI::CloseWindowT<ListHeaviestUI>();
}

void MainUI::InitCallback() {
	callback.add(Engine::CallbackType::PRESS_KEY, [](const Engine::CallbackEventPtr& callbackEventPtr) {
		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (key == Engine::VirtualKey::F1) {
			if (UI::ShowingWindow<ComputationsUI>()) {
				UI::CloseWindowT<ComputationsUI>();
			}
			else {
				UI::ShowWindow<ComputationsUI>(MainUI::_mySystem);
			}
		}

		if (key == Engine::VirtualKey::F2) {
			if (UI::ShowingWindow<SpaceManagerUI>()) {
				UI::CloseWindowT<SpaceManagerUI>();
			}
			else {
				UI::ShowWindow<SpaceManagerUI>(MainUI::_mySystem);
			}
		}

		if (key == Engine::VirtualKey::F3) {
			if (UI::ShowingWindow<ListHeaviestUI>()) {
				UI::CloseWindowT<ListHeaviestUI>();
			}
			else {
				UI::ShowWindow<ListHeaviestUI>(MainUI::_mySystem);
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

		// �������� ������
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

	callback.add(Engine::CallbackType::RELEASE_TAP, [](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (IsLockAction()) {
			return;
		}

	if (MySystem::currentSpace;  Engine::TapCallbackEvent * tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
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
		}
	});
}

void MainUI::DrawOnSpace() {
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
}

bool MainUI::IsLockAction() {
	return ImGui::GetIO().WantCaptureMouse;
}

unsigned int MainUI::GetViewType() {
	if (BottomUI* bottomUI = dynamic_cast<BottomUI*>(UI::GetWindow<BottomUI>().get())) {
		return (unsigned int)bottomUI->_viewType;
	}
}
