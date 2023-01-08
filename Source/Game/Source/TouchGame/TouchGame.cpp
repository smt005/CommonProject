
#include "TouchGame.h"
#include "Core.h"
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"
#include "Screen.h"
#include "Common/Help.h"
#include "Draw/Camera/CameraControl.h"
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
#include "Maps/MenuMap.h"

#include  "ImGuiManager/Editor/EditorModel.h"
#include  "ImGuiManager/Editor/EditMap.h"
#include  "ImGuiManager/Editor/Console.h"

#include <memory>
#include <set>


#define DRAW DrawLight

std::string TouchGame::_resourcesDir;
const std::string saveFileName("../../../Executable/Save.json");
Object lightObject;
vec3 lightPos(-500.0f, -500.0f, 500.0f);

TouchGame::TouchGame() {
}

TouchGame::~TouchGame() {
	if (_text) {
		delete _text;
		_text = nullptr;
	}

	_callbackPtr.reset();
	RemoveGreed();
}

void TouchGame::init() {
	load();

	DRAW::setClearColor(0.3f, 0.6f, 0.9f, 1.0f);

	//...
	MenuMap::Ptr menuMap(new MenuMap());
	menuMap->create("Menu");
	Map::AddCurrentMap(Map::add(menuMap));

	MenuOrtoMap::Ptr menuOrthoMap(new MenuOrtoMap());
	menuOrthoMap->create("MenuOrto");
	Map::AddCurrentMap(Map::add(menuOrthoMap));

	//...
	initPhysic();
	update();
	initCallback();
}

void TouchGame::close() {
	save();
}

void TouchGame::update() {
	Map& map = Map::GetFirstCurrentMap();

	map.action();

	Engine::Physics::updateScene(Engine::Core::deltaTime() * 10.f);
	map.updatePhysixs();
}

void TouchGame::draw() {
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

	// MapEditor
	if (Object::Ptr object = Editor::MapEditor::NewObject()) {
		DRAW::draw(*object);
	}

	// Text
	DrawText();

}

void TouchGame::Drawline() {
	if (_greed) {
		Camera::Set<Camera>(Map::GetFirstCurrentMap().getCamera());
		DrawLine::prepare();
		DrawLine::draw(*_greed);
	}
}

void TouchGame::DrawText() {
	if (_text) {
		_text->Draw();
	} else {
		_text = new Engine::Text("123", "Fonts/tahoma.ttf");
		//_text->SavePNG();
	}
}

void TouchGame::resize() {
	Camera::GetLink().Resize();
}

void TouchGame::CheckMouse() {
	glm::vec2 mousePos = Engine::Callback::mousePos();

	if (mousePos.x < 0.f) {
		mousePos.x = Engine::Screen::width() - mousePos.x;
		Engine::Core::SetCursorPos(mousePos.x, mousePos.y);
	}
	else if (mousePos.x > Engine::Screen::width()) {
		mousePos.x = mousePos.x - Engine::Screen::width();
		Engine::Core::SetCursorPos(mousePos.x, mousePos.y);
	}

	if (mousePos.y < 0.f) {
		mousePos.y = Engine::Screen::height() - mousePos.y;
		Engine::Core::SetCursorPos(mousePos.x, mousePos.y);
	}
	else if (mousePos.y > Engine::Screen::height()) {
		mousePos.y = mousePos.y - Engine::Screen::height();
		Engine::Core::SetCursorPos(mousePos.x, mousePos.y);
	}
}

void TouchGame::initCallback() {
	_callbackPtr = std::make_shared<Engine::Callback>(Engine::CallbackType::PINCH_TAP, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (UI::ShowingWindow("Edit model") || Editor::Console::IsLock()) {
			return;
		}

		if (Engine::Callback::pressTap(Engine::VirtualTap::LEFT)) {
			if (UI::ShowingWindow("Edit model")) {
				return;
			}

			/* NEW_CAMERA
			glm::vec3 cursorPos3 = Camera::current.corsorCoord();
			Object& object = Map::GetFirstCurrentMap().getObjectByName("Cylinder");
			glm::vec3 pos3 = object.getPos();
			glm::vec3 forceVec3 = cursorPos3 - pos3;
			forceVec3 = glm::normalize(forceVec3);
			forceVec3 *= _force;
			object.addForce(forceVec3);*/
		}
	});

	_callbackPtr->add(Engine::CallbackType::SCROLL, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::TapCallbackEvent * tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			if (UI::ShowingWindow("Edit model") || Editor::Console::IsLock()) {
				return;
			}
			/* NEW_CAMERA
			if (tapCallbackEvent->_id == Engine::VirtualTap::SCROLL_UP) {
				float fov = Camera::current.fov();
				fov -= 0.1f;
				if (fov >= 44.1f) {
					Camera::current.setFov(fov);
				}
			}
			else if (tapCallbackEvent->_id == Engine::VirtualTap::SCROLL_BOTTOM) {
				float fov = Camera::current.fov();
				fov += 0.1f;
				if (fov <= 46.5f) {
					Camera::current.setFov(fov);
				}
			}*/
		}
		});

	_callbackPtr->add(Engine::CallbackType::PRESS_TAP, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Editor::Console::IsLock()) {
			return;
		}

		//if (Engine::Callback::pressTap(Engine::VirtualTap::LEFT)) {
			if (Object::Ptr newObject = Editor::MapEditor::NewObject()) {
				Editor::MapEditor::AddObjectToMap();
			}
		//}
	});

	_callbackPtr->add(Engine::CallbackType::RELEASE_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Editor::Console::IsLock()) {
			return;
		}

		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (key == Engine::VirtualKey::ESCAPE) {
			Map::Ptr& map = Map::SetCurrentMap(Map::getByName("Menu"));
			// NEW_CAMERA Camera::setCurrent(map->getCamera());
			//Engine::Core::close();
		}
	});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (UI::ShowingWindow("Edit model") || Editor::Console::IsLock()) {
			return;
		}
		//...
	});

	_callbackPtr->add(Engine::CallbackType::MOVE, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (UI::ShowingWindow("Edit model") || Editor::Console::IsLock()) {
			return;
		}
		//...
	});

	_callbackPtr->add(Engine::CallbackType::RELEASE_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (key == Engine::VirtualKey::TILDE) {
			if (UI::ShowingWindow("Console")) {
				UI::CloseWindow("Console");
			}
			else {
				UI::ShowWindow<Editor::Console>();
			}
		}

		if (key == Engine::VirtualKey::F1) {
			if (UI::ShowingWindow("Edit model")) {
				UI::CloseWindow("Edit model");
			}
			else {
				UI::ShowWindow<Editor::ModelEditor>();
			}
		}
		if (key == Engine::VirtualKey::F2) {
			if (UI::ShowingWindow("Edit map")) {
				UI::CloseWindow("Edit map");
			}
			else {
				UI::ShowWindow<Editor::MapEditor>();
			}
		}

		//...
		if (key == Engine::VirtualKey::G && Engine::Callback::pressKey(Engine::VirtualKey::CONTROL)) {
			if (_greed) {
				RemoveGreed();
			} else {
				MakeGreed();
			}
		}

		if (key == Engine::VirtualKey::S && Engine::Callback::pressKey(Engine::VirtualKey::CONTROL)) {
			save();
		}

		if (key == Engine::VirtualKey::L && Engine::Callback::pressKey(Engine::VirtualKey::CONTROL)) {
			loadCamera();
		}

		if (UI::WindowDisplayed() || Editor::Console::IsLock()) {
			return;
		}

		if (key == Engine::VirtualKey::SPACE) {
			Object& object = Map::GetFirstCurrentMap().getObjectByName("Cylinder");
			glm::vec3 pos3 = {0.f, 0.f , 10.f};
			object.setActorPos(pos3);
		}
	});
}

void TouchGame::initPhysic() {
	Engine::Physics::init();
	Engine::Physics::createScene();
	Map::GetFirstCurrentMap().initPhysixs();

}

bool TouchGame::load()
{
	Json::Value loadData;
	if (!help::loadJson(saveFileName, loadData) || loadData.empty()) {
		return false;
	}

	//...
	if (loadData["Greed"].isBool() && loadData["Greed"].asBool()) {
		MakeGreed();
	} else {
		RemoveGreed();
	}

#if _DEBUG
	Engine::Core::log("LOAD: " + saveFileName + "\n" + help::stringFroJson(loadData));
#endif // _DEBUG

	return true;
}

bool TouchGame::loadCamera() {
	Json::Value loadData;
	if (!help::loadJson(saveFileName, loadData) || loadData.empty()) {
		return false;
	}

	if (!loadData["Camera"].empty()) {
		if (CameraControl* cameraPtr = dynamic_cast<CameraControl*>(Map::GetFirstCurrentMap().getCamera().get())) {
			cameraPtr->Load(loadData["Camera"]);
			cameraPtr->Enable(true);
		}
	}

#if _DEBUG
	Engine::Core::log("LOAD_CAMERA: " + saveFileName + "\n" + help::stringFroJson(loadData));
#endif // _DEBUG
}

void TouchGame::save()
{
	Json::Value saveData;

	if (CameraControl* cameraPtr = dynamic_cast<CameraControl*>(Map::GetFirstCurrentMap().getCamera().get())) {
		cameraPtr->Save(saveData["Camera"]);
	}

	if (_greed) {
		saveData["Greed"] = true;
	}

	//...
	help::saveJson(saveFileName, saveData);

#if _DEBUG
	Engine::Core::log("SAVE: " + saveFileName + "\n" + help::stringFroJson(saveData));
#endif // _DEBUG
}

void TouchGame::MakeGreed() {
	_greed = new Greed(100.0f, 10.0f);
	_greed->setPos({ 0.0f, 0.0f, 0.1f });
}

void TouchGame::RemoveGreed() {
	if (_greed) {
		delete _greed;
		_greed = nullptr;
	}
}
