#include "MainUI.h"
#include "imgui.h"
#include "TopUI.h"
#include "BottomUI.h"
#include "SpaceManagerUI.h"
#include "ListHeaviestUI.h"
#include "Draw/Camera/CameraControlOutside.h"
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"
#include "Draw/DrawLight.h"
#include "MySystem/MySystem.h"
#include "../Objects/SpaceManager.h"
#include "../Objects/Body.h"
#include "../Objects/Space.h"
#include "Object/Object.h"
#include "Object/Model.h"

namespace {
	Engine::Callback callback;
	std::shared_ptr<Space> spacePtr;
	Object::Ptr bodyMarker;
}

MySystem* MainUI::_mySystem = nullptr;

void MainUI::Open(MySystem* mySystem) {
	_mySystem = mySystem;
	if (!_mySystem || !_mySystem->_space) {
		return;
	}

	spacePtr = _mySystem->_space;

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

		if (key == Engine::VirtualKey::W) {
			if (UI::ShowingWindow<SpaceManagerUI>()) {
				UI::CloseWindowT<SpaceManagerUI>();
			}
			else {
				UI::ShowWindow<SpaceManagerUI>(MainUI::_mySystem);
			}
		}

		if (key == Engine::VirtualKey::E) {
			if (UI::ShowingWindow<ListHeaviestUI>()) {
				UI::CloseWindowT<ListHeaviestUI>();
			}
			else {
				UI::ShowWindow<ListHeaviestUI>(MainUI::_mySystem);
			}
		}
	});

	callback.add(Engine::CallbackType::RELEASE_TAP, [](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (IsLockAction()) {
			return;
		}

		if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			if (tapCallbackEvent->_id == Engine::VirtualTap::MIDDLE) {
				const glm::mat4x4& matCamera = _mySystem->_camearCurrent->ProjectView();
				spacePtr->_focusBody = spacePtr->HitObject(matCamera);
				spacePtr->_selectBody = spacePtr->_focusBody;
			}

			if (tapCallbackEvent->_id == Engine::VirtualTap::LEFT) {
				if (BottomUI* bottomUI = dynamic_cast<BottomUI*>(UI::GetWindow<BottomUI>().get())) {
					if (bottomUI->_addBodyType == AddBodyType::ORBIT) {
						auto cursorPosGlm = _mySystem->_camearCurrent->corsorCoord();
						Math::Vector3d cursorPos(cursorPosGlm.x, cursorPosGlm.y, cursorPosGlm.z);
						SpaceManager::AddObjectOnOrbit(spacePtr.get(), cursorPos);
						bottomUI->_addBodyType = AddBodyType::NONE;
					}
				}
			}
		}
	});
}

void MainUI::DrawOnSpace() {
	if (!bodyMarker) {
		bodyMarker = std::make_shared<Object>("Marker", "Marker");
	}

	const glm::mat4x4& matCamera = _mySystem->_camearCurrent->ProjectView();

	Camera::Set(_mySystem->_camearScreen);
	DrawLight::prepare();

	for (Body::Ptr& body : spacePtr->_bodies) {	
		Math::Vector3 posOnScreen = body->PosOnScreen(matCamera, false);
		float pos[] = { posOnScreen.x, posOnScreen.y, posOnScreen.z };
		
		bodyMarker->setPos(pos);
		DrawLight::draw(*bodyMarker);
	}
}

bool MainUI::IsLockAction() {
	return ImGui::GetIO().WantCaptureMouse;
}

unsigned int MainUI::GetViewType() {
	if (BottomUI* bottomUI = dynamic_cast<BottomUI*>(UI::GetWindow<BottomUI>().get())) {
		return (unsigned int)bottomUI->_viewType;
	}
}