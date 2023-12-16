#include "MainUI.h"

#include "TopUI.h"
#include "BottomUI.h"
#include "SystemManager.h"
#include "ListHeaviestUI.h"
#include "Draw/Camera/CameraControlOutside.h"
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"

#include "SystemMy/SystemMy.h"
#include "SystemMy/Objects/SystemMapMyShared.h"

//#include "iostream"

namespace {
	Engine::Callback callback;
	std::shared_ptr<SystemMap> spacePtr;
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
		if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			if (tapCallbackEvent->_id == Engine::VirtualTap::LEFT) {
				if (Body::Ptr body = spacePtr->HitObject()) {
					spacePtr->_focusBody = body;
				}
			}
		}
	});
}
