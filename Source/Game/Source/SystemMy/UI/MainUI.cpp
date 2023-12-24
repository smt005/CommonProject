#include "MainUI.h"
#include "imgui.h"
#include "TopUI.h"
#include "BottomUI.h"
#include "SystemManager.h"
#include "ListHeaviestUI.h"
#include "Draw/Camera/CameraControlOutside.h"
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"
#include "Draw/DrawLight.h"
#include "../Objects/SpaceManager.h"
#include "SystemMy/SystemMy.h"
#include "SystemMy/Objects/SystemMapMyShared.h"
#include "Object/Object.h"
#include "Object/Model.h"

namespace {
	Engine::Callback callback;
	std::shared_ptr<SystemMap> spacePtr;
	Object::Ptr bodyMarker;
}

SystemMy* MainUI::systemMy = nullptr;

void MainUI::Open(SystemMy* systemMyArg) {
	systemMy = systemMyArg;
	if (!systemMy || !systemMy->_systemMap) {
		return;
	}

	spacePtr = systemMy->_systemMap;

	InitCallback();

	UI::ShowWindow<TopUI>(systemMy);
	UI::ShowWindow<BottomUI>(systemMy);
}

void MainUI::Hide() {
	UI::CloseWindowT<SystemManager>();
	UI::CloseWindowT<ListHeaviestUI>();
}

void MainUI::InitCallback() {
	callback.add(Engine::CallbackType::PRESS_KEY, [](const Engine::CallbackEventPtr& callbackEventPtr) {
		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (key == Engine::VirtualKey::W) {
			if (UI::ShowingWindow<SystemManager>()) {
				UI::CloseWindowT<SystemManager>();
			}
			else {
				UI::ShowWindow<SystemManager>(MainUI::systemMy);
			}
		}

		if (key == Engine::VirtualKey::E) {
			if (UI::ShowingWindow<ListHeaviestUI>()) {
				UI::CloseWindowT<ListHeaviestUI>();
			}
			else {
				UI::ShowWindow<ListHeaviestUI>(MainUI::systemMy);
			}
		}
	});

	callback.add(Engine::CallbackType::RELEASE_TAP, [](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (IsLockAction()) {
			return;
		}

		if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			if (tapCallbackEvent->_id == Engine::VirtualTap::MIDDLE) {
				const glm::mat4x4& matCamera = systemMy->_camearCurrent->ProjectView();
				spacePtr->_focusBody = spacePtr->HitObject(matCamera);
				spacePtr->_selectBody = spacePtr->_focusBody;
			}

			if (tapCallbackEvent->_id == Engine::VirtualTap::LEFT) {
				if (BottomUI* bottomUI = dynamic_cast<BottomUI*>(UI::GetWindow<BottomUI>().get())) {
					if (bottomUI->_addBodyType == AddBodyType::ORBIT) {
						auto cursorPosGlm = systemMy->_camearCurrent->corsorCoord();
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

	const glm::mat4x4& matCamera = systemMy->_camearCurrent->ProjectView();

	Camera::Set(systemMy->_camearScreen);
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
