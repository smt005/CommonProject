
#include "System.h"
#include "Core.h"
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"
#include "Screen.h"
#include "Common/Help.h"
//#include "Draw/Camera/CameraControl.h"
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
#include "Physics/Physics.h"

#include "Objects//Body.h"

#define DRAW DrawLight
std::string System::_resourcesDir;
const std::string saveFileName("../../../Executable/Save.json");

System::System() {
}

System::~System() {;
}

void System::init() {
	DRAW::setClearColor(0.3f, 0.6f, 0.9f, 1.0f);

	Map::Ptr mainMap(new Map());
	Map::AddCurrentMap(Map::add(mainMap));

	//...
	if (!load()) {
		mainMap->getCamera() = std::make_shared<CameraControlOutside>();
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
	InitPhysic();

	//...
	{
		Body::system = mainMap;
		
		Body* sunObject = new Body("Sun", "OrangeStar", { 0.1f, 0.1f, 0.0f });
		sunObject->createActorPhysics();
		sunObject->setMass(10000.f);
		sunObject->LinePath().color = { 0.9f, 0.1f, 0.1f, 0.5f };
		mainMap->addObject(sunObject);

		_suns.emplace_back(sunObject);
	}

	// Interface
	{
		Object& obj = mainMap->addObjectToPos("Plane", { 2.f, 3.f, 0.f });
		obj.setName("Target");
	}
	{
		Object& obj = mainMap->addObjectToPos("Plane", { 14.f, 12.f, 0.f });
		obj.setName("Vector");
	}
}

void System::InitPhysic() {
	Engine::Physics::init();
	Engine::Physics::GetGravity({0.f, 0.f, 0.f});
	Engine::Physics::createScene();
	Map::GetFirstCurrentMap().initPhysixs();

	for (auto& objPtr : Map::GetFirstCurrentMap().GetObjects()) {
		objPtr->setMass(objPtr->mass);
	}

}

void System::close() {
	save();
}

void System::update() {
	Body::RemoveBody();

	Map& map = Map::GetFirstCurrentMap();
	map.action();

	Engine::Physics::updateScene(Engine::Core::deltaTime() * 1.f);
	map.updatePhysixs();

	Body::CenterSystem();
	Body::CenterMassSystem();
}

void System::draw() {
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

void System::Drawline() {
	Camera::Set<Camera>(Map::GetFirstCurrentMap().getCamera());
	DrawLine::prepare();

	if (_greed) {
		DrawLine::draw(*_greed);
	}
	else {
		_greed = new Greed(10000.0f, 1000.0f);
		_greed->setPos({ 0.0f, 0.0f, 0.1f });
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
			if (!objPtr->hasPhysics() && objPtr->tag != 123) {
				continue;
			}

			Body* body = static_cast<Body*>(objPtr.get());
			DrawLine::draw(body->LineToCenter());
		}
	}
	if (showCenterMass) {
		for (auto& objPtr : Map::GetFirstCurrentMap().GetObjects()) {
			if (!objPtr->hasPhysics() && objPtr->tag != 123) {
				continue;
			}

			Body* body = static_cast<Body*>(objPtr.get());
			DrawLine::draw(body->LineToMassCenter());
		}
	}
	if (showForceVector) {
		for (auto& objPtr : Map::GetFirstCurrentMap().GetObjects()) {
			if (!objPtr->hasPhysics() && objPtr->tag != 123) {
				continue;
			}

			Body* body = static_cast<Body*>(objPtr.get());
			DrawLine::draw(body->LineForceVector());
		}
	}
	if (showPath) {
		for (auto& objPtr : Map::GetFirstCurrentMap().GetObjects()) {
			if (!objPtr->hasPhysics() && objPtr->tag != 123) {
				continue;
			}

			Body* body = static_cast<Body*>(objPtr.get());
			DrawLine::draw(body->LinePath());
		}
	}
}

void System::resize() {
	Camera::GetLink().Resize();
}

bool System::load() {
	//return false; // TEMP_ .................................................
	Json::Value loadData;
	if (!help::loadJson(saveFileName, loadData) || loadData.empty()) {
		return false;
	}

	if (!loadData["Camera"].empty()) {
		Map::GetFirstCurrentMap().getCamera() = std::make_shared<CameraControlOutside>();
		if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())) {
			cameraPtr->Load(loadData["Camera"]);
			cameraPtr->Enable(true);
		}
	}


#if _DEBUG
	Engine::Core::log("LOAD: " + saveFileName + "\n" + help::stringFroJson(loadData));
#endif // _DEBUG
}

void System::save() {
	//return; // TEMP_ .................................................
	Json::Value saveData;

	if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())) {
		cameraPtr->Save(saveData["Camera"]);
	}

	//...
	help::saveJson(saveFileName, saveData);

#if _DEBUG
	Engine::Core::log("SAVE: " + saveFileName + "\n" + help::stringFroJson(saveData));
#endif // _DEBUG
}

void System::initCallback() {
	_callbackPtr = std::make_shared<Engine::Callback>(Engine::CallbackType::PRESS_TAP, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			if (tapCallbackEvent->_id == Engine::VirtualTap::LEFT) {
				if (_points.empty()) {
					_points.emplace_back(Map::GetFirstCurrentMap().getCamera()->corsorCoord());
					Map::GetFirstCurrentMap().getObjectByName("Target").setPos(_points[0]);
				}
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
		}
	});

	_callbackPtr->add(Engine::CallbackType::RELEASE_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (Engine::Callback::pressKey(Engine::VirtualKey::SPACE)) {
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
		}

		//...
		if (key == Engine::VirtualKey::Z) {
			--_curentSunn;
			if (_curentSunn >= _suns.size()) {
				_curentSunn = _suns.size() - 1;
			}
		}
		if (key == Engine::VirtualKey::X) {
			++_curentSunn;
			if (_curentSunn >= _suns.size()) {
				_curentSunn = 0;
			}
		}
	});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::Callback::pressKey(Engine::VirtualKey::Q) && Engine::Callback::pressKey(Engine::VirtualKey::SPACE)) {
			for (int i = 0; i < 10; ++i) {
				update();
			}
		}
	});

	_callbackPtr->add(Engine::CallbackType::RELEASE_TAP, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (_points.size() == 2) {
			std::string name = "Body_" + std::to_string(Map::GetFirstCurrentMap().GetObjects().size());
			Body* object = nullptr;
			
			if (Engine::Callback::pressKey(Engine::VirtualKey::SPACE)) {
				object = new Body(name, "OrangeStar", _points[0]);
				object->LinePath().color = { 0.9f, 0.1f, 0.1f, 0.5f };
				object->createActorPhysics();
				object->setMass(help::random(1000.f, 10000.f));

				_suns.emplace_back(object);
			} else {
				object = new Body(name, "BrownStone", _points[0]);
				object->LinePath().color = { 0.1f, 0.1f, 0.9f, 0.5f };
				object->createActorPhysics();
				object->setMass(help::random(0.1f, 1.f));
			}
			object->addForce((_points[0] - _points[1]) * object->mass);
			Map::GetFirstCurrentMap().addObject(object);

			_points.clear();
		}
	});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::Callback::pressKey(Engine::VirtualKey::SPACE)) {
			if (Engine::Callback::pressKey(Engine::VirtualKey::C)) {
				if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())) {
					cameraPtr->SetPosOutside(Body::centerSystem);
				}
			}

			if (Engine::Callback::pressKey(Engine::VirtualKey::S)) {
				if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(Map::GetFirstCurrentMap().getCamera().get())) {
					//auto& sun = Map::GetFirstCurrentMap().getObjectByName("Sun");
					//cameraPtr->SetPosOutside(sun.getPos());

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
				dist -= Engine::Callback::pressKey(Engine::VirtualKey::SHIFT) ? 50.f : 10.f;
			}
			else if (tapCallbackEvent->_id == Engine::VirtualTap::SCROLL_BOTTOM) {
				dist += Engine::Callback::pressKey(Engine::VirtualKey::SHIFT) ? 50.f : 10.f;
			}

			if (dist < 10.f) {
				dist = 10.f;
			}
			cameraPtr->SetDistanceOutside(dist);
		}
	});
}