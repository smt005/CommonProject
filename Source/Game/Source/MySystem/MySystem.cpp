
#include "MySystem.h"
#include "Core.h"
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"
#include "Screen.h"
#include "Common/Help.h"
#include "Draw/Camera/CameraControl.h"
#include "Draw/Camera/CameraControlOutside.h"
#include "Draw/Draw.h"
#include "Draw/DrawLight.h"
#include "Draw/DrawLine.h"
#include "Object/Line.h"
#include "ImGuiManager/UI.h"
#include "UI/MainUI.h"
#include "Math/Vector.h"

#include "Objects/BaseSpace.h"
#include "Objects/SpaceManager.h"
#include "Quests/Quest.h"

#define DRAW DrawLight
std::string MySystem::_resourcesDir;
const std::string saveFileName("../../../Executable/Save.json");

double minDt = std::numeric_limits<double>::max();
double maxDt = std::numeric_limits<double>::min();

MySystem::MySystem() {
}

MySystem::~MySystem() {
}

void MySystem::init() {
	//DRAW::setClearColor(0.3f, 0.6f, 0.9f, 1.0f);
	DRAW::setClearColor(0.1f, 0.2f, 0.3f, 1.0f);
	//DRAW::setClearColor(0.7f, 0.8f, 0.9f, 1.0f);

	//...
	_space = SpaceManager::Load("MAIN");
	_space->Load();

	//...
	_camearSide = std::make_shared<CameraControlOutside>();
	if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(_camearSide.get())) {
		cameraPtr->SetPerspective();
		cameraPtr->SetPos({ 0, 0, 0 });
		cameraPtr->SetDirect({ -0.440075815f, -0.717726171f, -0.539631844f });
		cameraPtr->SetSpeed(1.0);
		cameraPtr->SetDistanceOutside(50000.f);
		cameraPtr->Enable(true);
	}

	//...
	_camearTop = std::make_shared<Camera>();
	if (Camera* cameraPtr = dynamic_cast<Camera*>(_camearTop.get())) {
		cameraPtr->SetPerspective();
		cameraPtr->SetPos({ 0, 0, 100000 });
		cameraPtr->SetUp({ 0.f, -1.f, 0.f });
		cameraPtr->SetDirect({ 0.f, 0.f, -1.0f });
	}

	//...
	//_camearCurrent = _camearSide;
	_camearCurrent = _camearSide;
	Camera::Set<Camera>(_camearCurrent);

	//...
	_camearScreen = std::make_shared<Camera>(Camera::Type::ORTHO);

	//...
	initCallback();

	MainUI::Open(this);

	Quest::Load();
}

void MySystem::close() {
	save();
	MainUI::Hide();
}

void MySystem::update() {
	if ((Engine::Core::currentTime() - _time) < (_space->deltaTime > 33 ? 33 : _space->deltaTime)) {
		return;
	} else {
		_time = Engine::Core::currentTime();
	}

	_space->Update();

	Body& body = _space->RefFocusBody();
	auto centerMassT = body.GetPos();
	glm::vec3 centerMass = glm::vec3(centerMassT.x, centerMassT.y, centerMassT.z);
	if (auto camera = dynamic_cast<CameraControlOutside*>(_camearCurrent.get())) {
		camera->SetPosOutside(centerMass);
	} else if (auto camera = dynamic_cast<Camera*>(_camearCurrent.get())) {
		centerMass.z = 100000.f;
		camera->SetPos(centerMass);
	}
}

void MySystem::draw() {
	Camera::Set<Camera>(_camearCurrent);

	DRAW::viewport();
	DRAW::clearColor();
	DRAW::prepare();

	// SkyBox	
	if (MainUI::GetViewType() == 0 && _space->_skyboxObject) {
		auto camPos = Camera::_currentCameraPtr->Pos();
		_space->_skyboxObject->setPos(camPos);
		DRAW::draw(*_space->_skyboxObject);
		DRAW::clearDepth();
	}

	// Grid
	Drawline();

	// Draw
	DRAW::prepare();
	DRAW::DrawMap(*_space);
	DRAW::clearDepth();
	MainUI::DrawOnSpace();
}

void MySystem::Drawline() {
	DrawLine::prepare();

	if (_greed) {
		DrawLine::draw(*_greed);
	}
	else {
		_greed = new Greed(10000.0f, 1000.0f, { {0.5f, 0.25f, 0.25f, 0.125f}, { 0.125f, 0.125f, 0.5f, 0.125f }, { 0.75f, 1.f, 0.75f, 0.95f } });
		_greed->setPos({ 0.0f, 0.0f, 0.1f });
	}

	if (_greedBig) {
		DrawLine::draw(*_greedBig);
	}
	else {
		_greedBig = new Greed(100000.0f, 10000.0f, { {0.5f, 0.25f, 0.25f, 0.125f}, { 0.125f, 0.125f, 0.5f, 0.125f }, { 0.75f, 1.f, 0.75f, 0.95f } });
		_greedBig->setPos({ 0.0f, 0.0f, 0.1f });
	}

	if (_space) {
		/*if (_space->_skyboxModel) {
			DRAW::draw(*_space->_skyboxModel);
		}*/

		auto& star = _space->RefFocusBody();
		auto centerMassT = star.GetPos();
		glm::vec3 centerMass = glm::vec3(centerMassT.x, centerMassT.y, centerMassT.z);
		float cm[3] = { centerMass.x, centerMass.y, centerMass.z };
		DrawLine::SetIdentityMatrixByPos(cm);
		DrawLine::Draw(_space->spatialGrid);
	}
}

void MySystem::resize() {
	if (_camearSide) _camearSide->Resize();
	if (_camearTop) _camearTop->Resize();
	if (_camearScreen) _camearScreen->Resize();
}

bool MySystem::load() {
	Json::Value loadData;
	if (!help::loadJson(saveFileName, loadData) || loadData.empty()) {
		return false;
	}

#if _DEBUG
	Engine::Core::log("LOAD: " + saveFileName + "\n" + help::stringFroJson(loadData));
#endif // _DEBUG
}

void MySystem::save() {
	Json::Value saveData;

	//...
	help::saveJson(saveFileName, saveData);

#if _DEBUG
	Engine::Core::log("SAVE: " + saveFileName + "\n" + help::stringFroJson(saveData));
#endif // _DEBUG
}

void MySystem::initCallback() {
	_callbackPtr = std::make_shared<Engine::Callback>(Engine::CallbackType::PRESS_TAP, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		_lockMouse.lockPinch = Engine::Callback::mousePos().y > (Engine::Screen::height() - _lockMouse.bottomHeight);
		if (_lockMouse.IsLock()) {
			return;
		}


		if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			if (tapCallbackEvent->_id == Engine::VirtualTap::LEFT) {
				//...
			}
		}
	});

	_callbackPtr->add(Engine::CallbackType::MOVE, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		//...
	});

	_callbackPtr->add(Engine::CallbackType::RELEASE_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (key == Engine::VirtualKey::VK_1) {
			//...
		}
	});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		//...
	});

	_callbackPtr->add(Engine::CallbackType::RELEASE_TAP, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		/*if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			if (!MainUI::IsLockAction() && tapCallbackEvent->_id == Engine::VirtualTap::LEFT) {
				auto cursorPosGlm = _camearCurrent->corsorCoord();
				Math::Vector3d cursorPos(cursorPosGlm.x, cursorPosGlm.y, cursorPosGlm.z);
				SpaceManager::AddObjectOnOrbit(_space.get(), cursorPos);
			}
		}*/
	});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::Callback::pressKey(Engine::VirtualKey::SPACE)) {
			//...
		}
	});

	_callbackPtr->add(Engine::CallbackType::SCROLL, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(_camearCurrent.get());
			if (!cameraPtr) {
				return;
			}

			float dist = cameraPtr->GetDistanceOutside();

			if (tapCallbackEvent->_id == Engine::VirtualTap::SCROLL_UP) {
				dist -= Engine::Callback::pressKey(Engine::VirtualKey::SHIFT) ? 5000.f : 2000.f;
			}
			else if (tapCallbackEvent->_id == Engine::VirtualTap::SCROLL_BOTTOM) {
				dist += Engine::Callback::pressKey(Engine::VirtualKey::SHIFT) ? 5000.f : 2000.f;
			}

			if (dist < 10.f) {
				dist = 10.f;
			}
			cameraPtr->SetDistanceOutside(dist);
		}
	});
}
