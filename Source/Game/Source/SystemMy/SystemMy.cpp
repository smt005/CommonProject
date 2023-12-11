
#include "SystemMy.h"
#include "Core.h"
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"
#include "Screen.h"
#include "Common/Help.h"
#include "Draw/Camera/CameraControlOutside.h"
#include "Draw/Draw.h"
#include "Draw/DrawLight.h"
#include "Draw/DrawLine.h"
#include "Object/Map.h"
#include "Object/Object.h"
#include "Object/Model.h"
#include "Object/Shape.h"
#include "Object/Line.h"
#include "Object/Text.h"
#include "ImGuiManager/UI.h"
#include "glm/vec2.hpp"
#include "ImGuiManager/UI.h"

#include "UI/TopUI.h"
#include "UI/BottomUI.h"
#include "UI/ListHeaviestUI.h"

#include "UI/SystemManager.h"
#include "SaveManager.h"
#include "Math/Vector.h"
#include "Objects/SystemClass.h"
#include "Objects/SystemMapEasyMerger.h"
#include "Objects/SystemMapShared.h"
#include "Objects/SystemMapMyShared.h"

#define DRAW DrawLight
std::string SystemMy::_resourcesDir;
double SystemMy::time = 0;

const std::string saveFileName("../../../Executable/Save.json");

double minDt = std::numeric_limits<double>::max();
double maxDt = std::numeric_limits<double>::min();

SystemMy::SystemMy() {
}

SystemMy::~SystemMy() {;
}

void SystemMy::init() {
	//DRAW::setClearColor(0.3f, 0.6f, 0.9f, 1.0f);
	//DRAW::setClearColor(0.1f, 0.2f, 0.3f, 1.0f);
	DRAW::setClearColor(0.7f, 0.8f, 0.9f, 1.0f);

	//...
	_systemMap = std::shared_ptr<SystemMap>(new SystemMap("MAIN"));
	_systemMap->Load();

	//...
	if (!load()) {
		_camearSide = std::make_shared<CameraControlOutside>();
		Map::GetFirstCurrentMap().getCamera() = _camearSide;

		if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())) {
			cameraPtr->SetPerspective();
			cameraPtr->SetPos({ 10, 10, 10 });
			cameraPtr->SetDirect({ -0.440075815f, -0.717726171f, -0.539631844f });
			cameraPtr->SetSpeed(1.0);
			cameraPtr->Enable(true);
		}
	}

	//...
	initCallback();

	UI::ShowWindow<TopUI>(this);
	UI::ShowWindow<BottomUI>(this);

	// Interface
	{
		Object& obj = Map::GetFirstCurrentMap().addObjectToPos("DottedCircleTarget", { 2.f, 3.f, 0.f });
		obj.setName("Target");
		obj.setVisible(false);
	}
	{
		Object& obj = Map::GetFirstCurrentMap().addObjectToPos("Target", { 14.f, 12.f, 0.f });
		obj.setName("Vector");
		obj.setVisible(false);
	}
}

void SystemMy::close() {
	save();
}

void SystemMy::update() {
	if (time == 0) {
		time = Engine::Core::currentTime();
		return;
	}
	double dt = Engine::Core::currentTime() - time;

	if (dt > 0) {
		minDt = dt < minDt ? dt : minDt;
	}
	maxDt = dt > maxDt ? dt : maxDt;

	if (dt < 10) {
		return;
	}
	if (dt > 20) {
		dt = 20;
	}
	time = Engine::Core::currentTime();

	if (_timeSpeed == 0) {
		return;
	}

	if (_timeSpeed <= 100) {
		_systemMap->Update((double)_timeSpeed);
	} else {
		int day = _timeSpeed - 100;
		_systemMap->Update(100, day);
	}

	/*CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get());
	if (cameraPtr) {
		if (auto* starPtr = _systemMap->GetHeaviestBody(true)) {
			auto centerMassT = starPtr->GetPos();
			glm::vec3 centerMass = glm::vec3(centerMassT.x, centerMassT.y, centerMassT.z);
			cameraPtr->SetPosOutside(centerMass);
		}
	}*/
	
	Body& body = _systemMap->RefFocusBody();
	auto centerMassT = body.GetPos();
	glm::vec3 centerMass = glm::vec3(centerMassT.x, centerMassT.y, centerMassT.z);
	dynamic_cast<CameraControlOutside*>(_camearSide.get())->SetPosOutside(centerMass);

}

void SystemMy::draw() {
	Camera::Set<Camera>(_camearSide);

	DRAW::viewport();
	DRAW::clearColor();

	// Grid
	Drawline();

	// Draw SystemMap
	Camera::Set<Camera>(_camearSide);
	DRAW::prepare();
	DRAW::DrawMap(*_systemMap);
}

void SystemMy::Drawline() {
	//Camera::Set<Camera>(Map::GetFirstCurrentMap().getCamera());
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

	if (_systemMap) {
#if SYSTEM_MAP < 7
		auto* starPtr = _systemMap->GetHeaviestBody(true);
#else
		auto starPtr = _systemMap->GetHeaviestBody(true);
#endif
		if (starPtr) {
			auto centerMassT = starPtr->GetPos();
			glm::vec3 centerMass = glm::vec3(centerMassT.x, centerMassT.y, centerMassT.z);
			float cm[3] = { centerMass.x, centerMass.y, centerMass.z };
			DrawLine::SetIdentityMatrixByPos(cm);
			DrawLine::Draw(_systemMap->spatialGrid);
		}
	}

	if (_interfaceLine) {
		if (_points.size() == 2) {
			glm::vec3 posTarget = Map::GetFirstCurrentMap().getObjectByName("Target").getPos();
			glm::vec3 posVector = Map::GetFirstCurrentMap().getObjectByName("Vector").getPos();

			float points[] = { posTarget.x, posTarget.y, posTarget.z, posVector.x, posVector.y, posVector.z };
			_interfaceLine->set(points, 2);
			DrawLine::draw(*_interfaceLine);
		}
	}
	else {
		_interfaceLine = new Line();
		_interfaceLine->setType(0x0001);
	}

	//
	if (showCenter) {
		//...
	}
	if (showCenterMass) {
		//...
	}
	if (showForceVector) {
		//...
	}
	if (showPath) {
		//...
	}
	if (showRelativePath) {
		//...
	}
	
}

void SystemMy::resize() {
	Camera::GetLink().Resize();
}

bool SystemMy::load() {
	//return false; // TEMP_ .................................................
	Json::Value loadData;
	if (!help::loadJson(saveFileName, loadData) || loadData.empty()) {
		return false;
	}

	/*if (!loadData["Camera"].empty()) {
		//Map::GetFirstCurrentMap().getCamera() = std::make_shared<CameraControlOutside>();
		_camearSide = std::make_shared<CameraControlOutside>();
		Map::GetFirstCurrentMap().getCamera() = _camearSide;

		if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())) {
			cameraPtr->Load(loadData["Camera"]);
			cameraPtr->Enable(true);
			cameraPtr->SetDistanceOutside(5000.f);
		}
	}*/

	if (!loadData["CameraSide"].empty()) {
		_camearSide = std::make_shared<CameraControlOutside>();
		
		dynamic_cast<CameraControlOutside*>(_camearSide.get())->Load(loadData["CameraSide"]);
		dynamic_cast<CameraControlOutside*>(_camearSide.get())->Enable(true);
		dynamic_cast<CameraControlOutside*>(_camearSide.get())->SetDistanceOutside(50000.f);

		Map::GetFirstCurrentMap().getCamera() = _camearSide;
	}

	if (!loadData["CamearTop"].empty()) {
		_camearTop = std::make_shared<CameraControlOutside>();

		dynamic_cast<CameraControlOutside*>(_camearTop.get())->Load(loadData["CamearTop"]);
		dynamic_cast<CameraControlOutside*>(_camearTop.get())->Enable(true);
		dynamic_cast<CameraControlOutside*>(_camearTop.get())->SetDistanceOutside(5000.f);
	}

#if _DEBUG
	Engine::Core::log("LOAD: " + saveFileName + "\n" + help::stringFroJson(loadData));
#endif // _DEBUG
}

void SystemMy::save() {
	//return; // TEMP_ .................................................
	Json::Value saveData;

	//if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())) {
		//cameraPtr->Save(saveData["Camera"]);
	//}

	if (_camearSide) {
		_camearSide->Save(saveData["CameraSide"]);
	}
	if (_camearTop) {
		_camearTop->Save(saveData["CamearTop"]);
	}

	//...
	help::saveJson(saveFileName, saveData);

#if _DEBUG
	Engine::Core::log("SAVE: " + saveFileName + "\n" + help::stringFroJson(saveData));
#endif // _DEBUG
}

void SystemMy::initCallback() {
	_callbackPtr = std::make_shared<Engine::Callback>(Engine::CallbackType::PRESS_TAP, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		_lockMouse.lockPinch = Engine::Callback::mousePos().y > (Engine::Screen::height() - _lockMouse.bottomHeight);
		if (_lockMouse.IsLock()) {
			return;
		}


		if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			if (tapCallbackEvent->_id == Engine::VirtualTap::LEFT) {
				if (_points.empty()) {
					_points.emplace_back(Map::GetFirstCurrentMap().getCamera()->corsorCoord());
					Map::GetFirstCurrentMap().getObjectByName("Target").setPos(_points[0]);
					Map::GetFirstCurrentMap().getObjectByName("Target").setVisible(true);
				}
			}
		}
	});

	_callbackPtr->add(Engine::CallbackType::PRESS_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (key == Engine::VirtualKey::W) {
			if (UI::ShowingWindow<SystemManager>()) {
				UI::CloseWindowT<SystemManager>();
				_lockMouse.lockAllPinch = false;
			}
			else {
				UI::ShowWindow<SystemManager>(this);
				_lockMouse.lockAllPinch = true;
			}
		}

		if (key == Engine::VirtualKey::E) {
			if (UI::ShowingWindow<ListHeaviestUI>()) {
				UI::CloseWindowT<ListHeaviestUI>();
				_lockMouse.lockAllPinch = false;
			}
			else {
				UI::ShowWindow<ListHeaviestUI>(this);
				_lockMouse.lockAllPinch = true;
			}
		}
	});

	_callbackPtr->add(Engine::CallbackType::MOVE, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (_points.size() == 1) {
			_points.emplace_back(Map::GetFirstCurrentMap().getCamera()->corsorCoord());
		}
		if (_points.size() == 2) {
			_points[1] = Map::GetFirstCurrentMap().getCamera()->corsorCoord();
			Map::GetFirstCurrentMap().getObjectByName("Vector").setPos(_points[1]);
			Map::GetFirstCurrentMap().getObjectByName("Vector").setVisible(true);
		}
	});

	_callbackPtr->add(Engine::CallbackType::RELEASE_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		//if (Engine::Callback::pressKey(Engine::VirtualKey::SPACE)) {}

		if (key == Engine::VirtualKey::V) {
			if (_camearTop != Map::GetFirstCurrentMap().getCamera()) {
				if (!_camearTop) {
					_camearTop = std::make_shared<CameraControlOutside>();
					static_cast<CameraControlOutside*>(_camearTop.get())->SetPerspective();
					static_cast<CameraControlOutside*>(_camearTop.get())->SetPos({ 1.f, 1.f, 10.f });
					static_cast<CameraControlOutside*>(_camearTop.get())->SetDirect({ -1.f, -1.f, 0.1f });
					static_cast<CameraControlOutside*>(_camearTop.get())->SetSpeed(1.0);
					static_cast<CameraControlOutside*>(_camearTop.get())->Enable(true);
				}

				static_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())->enableCallback = false;
				static_cast<CameraControlOutside*>(_camearTop.get())->enableCallback = true;
				Map::GetFirstCurrentMap().getCamera() = _camearTop;

				//DRAW::setClearColor(0.7f, 0.8f, 0.9f, 1.0f);
			} else {
				static_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())->enableCallback = false;
				static_cast<CameraControlOutside*>(_camearSide.get())->enableCallback = true;
				Map::GetFirstCurrentMap().getCamera() = _camearSide;

				//DRAW::setClearColor(0.1f, 0.2f, 0.3f, 1.0f);
			}
		}

		if (key == Engine::VirtualKey::VK_1) {
			if (showCenterMass) {
				showCenterMass = false;
			}
			else {
				showCenterMass = true;
			}
		}
		if (key == Engine::VirtualKey::VK_2) {
			if (showCenter) {
				showCenter = false;
			}
			else {
				showCenter = true;
			}
		}
		if (key == Engine::VirtualKey::VK_3) {
			if (showForceVector) {
				showForceVector = false;
			}
			else {
				showForceVector = true;
			}
		}
		if (key == Engine::VirtualKey::VK_4) {
			if (showPath) {
				showPath = false;
			}
			else {
				showPath = true;
			}
		}
		if (key == Engine::VirtualKey::VK_5) {
			if (showRelativePath) {
				showRelativePath = false;
				//...
			}
			else {
				showRelativePath = true;

				//...
			}
		}

		//...
		if (key == Engine::VirtualKey::Z) {
			//...
		}
		if (key == Engine::VirtualKey::X) {
			//...
		}
	});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		/*if (Engine::Callback::pressKey(Engine::VirtualKey::Q) && Engine::Callback::pressKey(Engine::VirtualKey::SPACE)) {
			//...
		}*/
	});

	_callbackPtr->add(Engine::CallbackType::RELEASE_TAP, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (_points.size() == 2 || (_orbite == 0 && _points.size() == 1)) {
			if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
				if (tapCallbackEvent->_id == Engine::VirtualTap::LEFT) {
					float mass = 100;

					if (!_orbite) {
#if SYSTEM_MAP < 7
						auto* starPtr = _systemMap->GetHeaviestBody(true);
#else
						auto starPtr = _systemMap->GetHeaviestBody(true);
#endif
						if (starPtr) {
							auto starPosT = starPtr->GetPos();
							glm::vec3 gravityVector = _points[0] - glm::vec3(starPosT.x, starPosT.y, starPosT.z);
							glm::vec3 normalizeGravityVector = glm::normalize(gravityVector);

							float g90 = glm::pi<float>() / 2.0;
							glm::vec3 velocity(normalizeGravityVector.x* std::cos(g90) - normalizeGravityVector.y * std::sin(g90),
								normalizeGravityVector.x* std::sin(g90) + normalizeGravityVector.y * std::cos(g90),
								0.f);

							velocity *= std::sqrtf(_systemMap->_constGravity * starPtr->_mass / glm::length(gravityVector));
							_systemMap->Add("BrownStone", _points[0], velocity, mass, "");
						}
					} else {
						glm::vec3 velocity = _points[0] - _points[1];
						velocity /= 1000.f;
						_systemMap->Add("BrownStone", _points[0], velocity, mass, "");
					}

					_points.clear();
				}

				if (tapCallbackEvent->_id == Engine::VirtualTap::RIGHT) {
					//...
					_points.clear();
				}
			}
		}
	});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::Callback::pressKey(Engine::VirtualKey::SPACE)) {
			//...
		}
	});

	_callbackPtr->add(Engine::CallbackType::SCROLL, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get());
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

void SystemMy::SetPerspectiveView(bool perspectiveView) {
	_perspectiveView = perspectiveView;

	if (_perspectiveView) {
		static_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())->enableCallback = false;
		static_cast<CameraControlOutside*>(_camearSide.get())->enableCallback = true;
		Map::GetFirstCurrentMap().getCamera() = _camearSide;
	} else {
		if (!_camearTop) {
			_camearTop = std::make_shared<CameraControlOutside>();
			static_cast<CameraControlOutside*>(_camearTop.get())->SetPerspective();
			static_cast<CameraControlOutside*>(_camearTop.get())->SetPos({ 1.f, 1.f, 10.f });
			static_cast<CameraControlOutside*>(_camearTop.get())->SetDirect({ -1.f, -1.f, 0.1f });
			static_cast<CameraControlOutside*>(_camearTop.get())->SetSpeed(1.0);
			static_cast<CameraControlOutside*>(_camearTop.get())->Enable(true);
		}

		static_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())->enableCallback = false;
		static_cast<CameraControlOutside*>(_camearTop.get())->enableCallback = true;
		Map::GetFirstCurrentMap().getCamera() = _camearTop;
	}
}

void SystemMy::SetViewByObject(bool viewByObject) {
	_viewByObject = viewByObject;

	if (_viewByObject) {
		NormalizeSystem();
	}
}

void SystemMy::NormalizeSystem() {
	//...
}
