
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
#include "Objects/BodyMy.h"
#include "ImGuiManager/UI.h"
#include "UI/BottomUI.h"

#define DRAW DrawLight
std::string SystemMy::_resourcesDir;
double SystemMy::time = 0;

const std::string saveFileName("../../../Executable/Save.json");

SystemMy::SystemMy() {
}

SystemMy::~SystemMy() {;
}

void SystemMy::init() {
	//DRAW::setClearColor(0.3f, 0.6f, 0.9f, 1.0f);
	//DRAW::setClearColor(0.1f, 0.2f, 0.3f, 1.0f);
	DRAW::setClearColor(0.7f, 0.8f, 0.9f, 1.0f);

	Map::Ptr mainMap(new Map());
	Map::AddCurrentMap(Map::add(mainMap));

	//...
	if (!load()) {
		//mainMap->getCamera() = std::make_shared<CameraControlOutside>();
		_camearSide = std::make_shared<CameraControlOutside>();
		mainMap->getCamera() = _camearSide;

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
	UI::ShowWindow<BottomUI>(this);

	//...
	{
		BodyMy::system = mainMap;
		
		BodyMy* sunObject = new BodyMy("Sun", "OrangeStar", { 0.1f, 0.1f, 0.0f });
		sunObject->setMass(1000000.f);
		sunObject->LinePath().color = { 0.9f, 0.1f, 0.1f, 0.5f };
		mainMap->addObject(sunObject);

		_suns.emplace_back(sunObject);
	}

	// Interface
	{
		Object& obj = mainMap->addObjectToPos("DottedCircleTarget", { 2.f, 3.f, 0.f });
		obj.setName("Target");
		obj.setVisible(false);
	}
	{
		Object& obj = mainMap->addObjectToPos("Target", { 14.f, 12.f, 0.f });
		obj.setName("Vector");
		obj.setVisible(false);
	}
}

void SystemMy::close() {
	save();
}

void SystemMy::update() {
	if (_viewByObject && !_suns.empty()) {
		if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())) {
			if (!_suns.empty()) {
				auto sun = _suns[_curentSunn];
				cameraPtr->SetPosOutside(sun->getPos());
			}
		}
	}

	double dt = Engine::Core::currentTime() - time;
	if (dt < 10) {
		return;
	}
	time = Engine::Core::currentTime();

	Map& map = Map::GetFirstCurrentMap();

	map.action();
	BodyMy::ApplyForce(dt);

	for (int it = 1; it < _timeSpeed; ++it) {
		double dtTemp = 10;
		map.action();
		BodyMy::ApplyForce(dtTemp);
	}

	BodyMy::CenterSystem();
	BodyMy::CenterMassSystem();
	BodyMy::UpdateRalatovePos();
}

void SystemMy::draw() {
	DRAW::viewport();
	DRAW::clearColor();

	// Grid
	Drawline();

	// Draw
	for (Map::Ptr& map : Map::GetCurrentMaps()) {
		Camera::Set<Camera>(map->getCamera());
		DRAW::prepare();
		DRAW::draw(*map);
	}
}

void SystemMy::Drawline() {
	Camera::Set<Camera>(Map::GetFirstCurrentMap().getCamera());
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
		for (auto& objPtr : Map::GetFirstCurrentMap().GetObjects()) {
			if (objPtr->tag != 123) {
				continue;
			}

			BodyMy* body = static_cast<BodyMy*>(objPtr.get());
			DrawLine::draw(body->LineToCenter());
		}
	}
	if (showCenterMass) {
		for (auto& objPtr : Map::GetFirstCurrentMap().GetObjects()) {
			if (objPtr->tag != 123) {
				continue;
			}

			BodyMy* body = static_cast<BodyMy*>(objPtr.get());
			DrawLine::draw(body->LineToMassCenter());
		}
	}
	if (showForceVector) {
		for (auto& objPtr : Map::GetFirstCurrentMap().GetObjects()) {
			if (objPtr->tag != 123) {
				continue;
			}

			BodyMy* body = static_cast<BodyMy*>(objPtr.get());
			DrawLine::draw(body->LineForceVector());
		}
	}
	if (showPath) {
		for (auto& objPtr : Map::GetFirstCurrentMap().GetObjects()) {
			if (objPtr->tag != 123) {
				continue;
			}

			BodyMy* body = static_cast<BodyMy*>(objPtr.get());
			DrawLine::draw(body->LinePath());
		}
	}
	if (showRelativePath) {
		for (auto& objPtr : Map::GetFirstCurrentMap().GetObjects()) {
			if (objPtr->tag != 123) {
				continue;
			}

			BodyMy* body = static_cast<BodyMy*>(objPtr.get());
			DrawLine::draw(body->RelativelyLinePath());
		}
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
		dynamic_cast<CameraControlOutside*>(_camearSide.get())->SetDistanceOutside(5000.f);

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
		if (_lockMouse.lockPinch = Engine::Callback::mousePos().y > (Engine::Screen::height() - _lockMouse.bottomHeight)) {
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

		if (key == Engine::VirtualKey::P) {
			if (UI::ShowingWindow("BottomUI")) {
				UI::CloseWindow("BottomUI");
			}
			else {
				UI::ShowWindow<BottomUI>();
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
				BodyMy::centerBody = nullptr;
			}
			else {
				showRelativePath = true;

				if (!_suns.empty()) {
					auto sun = _suns[_curentSunn];
					BodyMy::centerBody = sun;
				}
			}
		}

		//...
		if (key == Engine::VirtualKey::Z) {
			--_curentSunn;
			if (_curentSunn >= _suns.size()) {
				_curentSunn = _suns.size() - 1;
			}

			if (showRelativePath) {
				if (!_suns.empty()) {
					auto sun = _suns[_curentSunn];
					BodyMy::centerBody = sun;
				}
			}
		}
		if (key == Engine::VirtualKey::X) {
			++_curentSunn;
			if (_curentSunn >= _suns.size()) {
				_curentSunn = 0;
			}

			if (showRelativePath) {
				if (!_suns.empty()) {
					auto sun = _suns[_curentSunn];
					BodyMy::centerBody = sun;
				}
			}
		}
	});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::Callback::pressKey(Engine::VirtualKey::Q) && Engine::Callback::pressKey(Engine::VirtualKey::SPACE)) {
			/*for (int i = 0; i < 100; ++i) {
				//update();

				double dt = 15;
				Map& map = Map::GetFirstCurrentMap();
				map.action();
				BodyMy::ApplyForce(dt);
			}*/
		}
	});

	_callbackPtr->add(Engine::CallbackType::RELEASE_TAP, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (_points.size() == 2) {
			if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
				if (tapCallbackEvent->_id == Engine::VirtualTap::LEFT) {
					std::string name = "Body_" + std::to_string(Map::GetFirstCurrentMap().GetObjects().size());
					BodyMy* object = nullptr;

					object = new BodyMy(name, "BrownStone", _points[0]);
					object->LinePath().color = { 0.1f, 0.1f, 0.9f, 0.5f };
					object->setMass(help::random(10.f, 100.f));

					//if (!Engine::Callback::pressKey(Engine::VirtualKey::SPACE)) {
					if (_orbite) {
						glm::vec3 velocity = _points[0] - _points[1];
						velocity /= 1000.f;
						object->_velocity = velocity;
					} else {
						glm::vec3 gravityVector = object->getPos() - _suns[_curentSunn]->getPos();
						glm::vec3 normalizeGravityVector = glm::normalize(gravityVector);

						float g90 = glm::pi<float>() / 2.0;
						glm::vec3 velocity(normalizeGravityVector.x * std::cos(g90) - normalizeGravityVector.y * std::sin(g90),
										normalizeGravityVector.x * std::sin(g90) + normalizeGravityVector.y * std::cos(g90),
										0.f);

						velocity *= std::sqrtf(BodyMy::G * _suns[_curentSunn]->mass / glm::length(gravityVector));
						object->_velocity = velocity;
					}

					Map::GetFirstCurrentMap().addObject(object);
					_points.clear();

					Map::GetFirstCurrentMap().getObjectByName("Target").setVisible(false);
					Map::GetFirstCurrentMap().getObjectByName("Vector").setVisible(false);
				}

				if (tapCallbackEvent->_id == Engine::VirtualTap::RIGHT) {
					std::string name = "Sun_" + std::to_string(Map::GetFirstCurrentMap().GetObjects().size());
					BodyMy* object = nullptr;

					object = new BodyMy(name, "OrangeStar", _points[0]);
					object->LinePath().color = { 0.9f, 0.1f, 0.1f, 0.5f };

					if (false) {
						object->setMass(help::random(500000.f, 1000000.f));
						object->AddForceMy((_points[0] - _points[1])* object->mass);
					} else {
						object->setMass(1000000.f);
						glm::vec3 velocity = _points[0] - _points[1];
						object->SetLinearVelocity(velocity);

						{
							if (_suns.size() == 1) {
								glm::vec3 velocityFirst = -velocity;
								_suns.front()->SetLinearVelocity(velocityFirst);
							}
						}
					}

					_suns.emplace_back(object);
					Map::GetFirstCurrentMap().addObject(object);

					_points.clear();
				}
			}
		}
	});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::Callback::pressKey(Engine::VirtualKey::SPACE)) {
			if (Engine::Callback::pressKey(Engine::VirtualKey::C)) {
				if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())) {
					cameraPtr->SetPosOutside(BodyMy::centerSystem);
				}
			}

			if (Engine::Callback::pressKey(Engine::VirtualKey::S)) {
				if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())) {
					if (!_suns.empty()) {
						auto sun = _suns[_curentSunn];
						cameraPtr->SetPosOutside(sun->getPos());
					}
				}
			}
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
				dist -= Engine::Callback::pressKey(Engine::VirtualKey::SHIFT) ? 250.f : 100.f;
			}
			else if (tapCallbackEvent->_id == Engine::VirtualTap::SCROLL_BOTTOM) {
				dist += Engine::Callback::pressKey(Engine::VirtualKey::SHIFT) ? 250.f : 100.f;
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
	glm::vec3 starVelocity = _suns[_curentSunn]->_velocity;
	glm::vec3 starPos = _suns[_curentSunn]->getPos();

	for (Object::Ptr& objectPtr : Map::GetFirstCurrentMap().GetObjects()) {
		if (objectPtr->tag != 123) {
			continue;
		}

		glm::vec3 velocity = static_cast<BodyMy*>(objectPtr.get())->_velocity;
		velocity -= starVelocity;
		static_cast<BodyMy*>(objectPtr.get())->_velocity = velocity;

		glm::vec3 pos = objectPtr->getPos();
		pos -= starPos;
		objectPtr->setPos(pos);

	}
}
