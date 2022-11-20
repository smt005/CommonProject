
#include "TouchGame.h"
#include "Core.h"
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"
#include "Screen.h"
#include "Common/Help.h"
#include "Draw/Camera.h"
//#include "Draw/Camera_Prototype_0/CameraTemp.h"
#include "Draw/Camera_Prototype_1/CameraProt2.h"
#include "Draw/Draw.h"
#include "Draw/DrawLight.h"
#include "Draw/DrawLine.h"
#include "Object/Map.h"
#include "Object/Object.h"
#include "Object/Model.h"
#include "Object/Shape.h"
#include "Object/Line.h"
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
	_callbackPtr.reset();

	if (_greed) {
		delete _greed;
		_greed = nullptr;
	}
}

void TouchGame::init() {
	_greed = new Greed(100.0f, 10.0f);
	_greed->setPos({ 0.0f, 0.0f, 0.1f });

	if (!load()) {
		Camera::current.setFromEye(true);
		Camera::current.setPos(glm::vec3(0.f, 95.f, 0.f));
		Camera::current.setVector(glm::vec3(0.f, -1.f, 0.f));
		Camera::current.setDist(1.0f);
	}

	DRAW::setClearColor(0.3f, 0.6f, 0.9f, 1.0f);

	bool typeMap = true;

	if (typeMap) {
		// MenuMap
		MenuMap::Ptr menuMap(new MenuMap());
		menuMap->create("Menu");
		Map::AddCurrentMap(Map::add(menuMap));
	} else {
		// MenuOrtoMap
		MenuOrtoMap::Ptr menuMap(new MenuOrtoMap());
		menuMap->create("MenuOrto");
		Map::AddCurrentMap(Map::add(menuMap));
	}

	initPhysic();
	update();
	initCallback();
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

	// Draw
	DRAW::prepare();
	for (Map::Ptr& map : Map::GetCurrentMaps()) {
		DRAW::draw(*map);
	}

	// MapEditor
	if (Object::Ptr object = Editor::MapEditor::NewObject()) {
		DRAW::draw(*object);
	}
}

void TouchGame::Drawline() {
	DrawLine::prepare();

	{
		float points[] = { 0.0f, 0.0f, 0.0f, 10.0f, 20.0f, 20.0f,
							10.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f };
		Line line(points, 4, Line::LINE);
		line.setLineWidth(5.0f);
		line.color = Color::GREEN;

		DrawLine::draw(line);
	}

	{
		float points[] = { 20.0f, 30.0f, 0.0f,
							20.0f, 30.0f, 20.0f,
							30.0f, 30.0f, 20.0f };
		Line line(points, 3, Line::LOOP);
		line.setLineWidth(5.0f);
		line.color = { 0.3f, 0.6f, 0.9f,0.5f };

		DrawLine::draw(line);
	}

	{
		float points[] = { 30.0f, 40.0f, 0.0f,
							30.0f, 40.0f, 20.0f,
							40.0f, 40.0f, 20.0f };
		Line line(points, 3, Line::STRIP);
		line.setLineWidth(5.0f);
		line.color = Color::RED;
		line.color.setAlpha(0.5);

		DrawLine::draw(line);
	}

	DrawLine::draw(*_greed);
}

void TouchGame::resize() {
	Camera::getCurrent().setPerspective(Camera::getCurrent().fov(), Engine::Screen::aspect(), 0.1f, 1000.0f);
	//CameraTemp::GetLink().Resize();
	CameraProt2::GetLink().Resize();
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

		if (Engine::Callback::pressTap(Engine::VirtualTap::RIGHT)) {
			Camera::current.rotate(Engine::Callback::deltaMousePos());
		}

		if (Engine::Callback::pressTap(Engine::VirtualTap::MIDDLE)) {
			Camera::current.move(Engine::Callback::deltaMousePos() * Engine::Core::deltaTime());
		}

		if (Engine::Callback::pressTap(Engine::VirtualTap::LEFT)) {
			if (UI::ShowingWindow("Edit model")) {
				return;
			}

			glm::vec3 cursorPos3 = Camera::current.corsorCoord();
			Object& object = Map::GetFirstCurrentMap().getObjectByName("Cylinder");
			glm::vec3 pos3 = object.getPos();
			glm::vec3 forceVec3 = cursorPos3 - pos3;
			forceVec3 = glm::normalize(forceVec3);
			forceVec3 *= _force;
			object.addForce(forceVec3);
		}
	});

	_callbackPtr->add(Engine::CallbackType::SCROLL, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::TapCallbackEvent * tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
			if (UI::ShowingWindow("Edit model") || Editor::Console::IsLock()) {
				return;
			}

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
			}
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
			Camera::setCurrent(map->getCamera());
			//Engine::Core::close();
		}
	});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (UI::ShowingWindow("Edit model") || Editor::Console::IsLock()) {
			return;
		}

		if (Engine::Callback::pressKey(Engine::VirtualKey::CONTROL)) {
			return;
		}

		// Камера
		float kForce = 1.0;
		if (Engine::Callback::pressKey(Engine::VirtualKey::SHIFT)) {
			kForce = 5.f;
		}

		if (Engine::Callback::pressKey(Engine::VirtualKey::S)) {
			Camera::current.move(CAMERA_FORVARD, kForce);
		}
		if (Engine::Callback::pressKey(Engine::VirtualKey::W)) {
			Camera::current.move(CAMERA_BACK, kForce);
		}
		if (Engine::Callback::pressKey(Engine::VirtualKey::D)) {
			Camera::current.move(CAMERA_RIGHT, kForce);
		}
		if (Engine::Callback::pressKey(Engine::VirtualKey::A)) {
			Camera::current.move(CAMERA_LEFT, kForce);
		}
		if (Engine::Callback::pressKey(Engine::VirtualKey::R)) {
			Camera::current.move(CAMERA_TOP, kForce);
		}
		if (Engine::Callback::pressKey(Engine::VirtualKey::F)) {
			Camera::current.move(CAMERA_DOWN, kForce);
		}
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

		if (key == Engine::VirtualKey::S && Engine::Callback::pressKey(Engine::VirtualKey::CONTROL)) {
			save();
		}

		if (key == Engine::VirtualKey::L && Engine::Callback::pressKey(Engine::VirtualKey::CONTROL)) {
			load();
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
	Json::Value saveData;
	if (!help::loadJson(saveFileName, saveData) || saveData.empty()) {
		return false;
	}

	if (!saveData["camera"].empty()) {
		Json::Value& cameraData = saveData["camera"];
		Camera::current.setJsonData(cameraData);
	}

	/*if (!saveData["cameraTemp"].empty()) {
		Json::Value& cameraData = saveData["cameraTemp"];
		CameraTemp::GetLink().Load(cameraData);
	}*/

	if (!saveData["CameraProt2"].empty()) {
		Json::Value& cameraData = saveData["CameraProt2"];
		CameraProt2::GetLink().Load(cameraData);
	}

#if _DEBUG
	Engine::Core::log("LOAD: " + saveFileName + "\n" + help::stringFroJson(saveData));
#endif // _DEBUG

	return true;
}

void TouchGame::save()
{
	Json::Value saveData;

	{
		Json::Value cameraData;
		Camera::current.getJsonData(cameraData);
		saveData["camera"] = cameraData;
	}

	/*{
		Json::Value cameraData;
		CameraTemp::GetLink().Save(cameraData);
		saveData["cameraTemp"] = cameraData;
	}*/

	{
		Json::Value cameraData;
		CameraProt2::GetLink().Save(cameraData);
		saveData["CameraProt2"] = cameraData;
	}

	saveData["testKey"] = "testValue";

	help::saveJson(saveFileName, saveData);

#if _DEBUG
	Engine::Core::log("SAVE: " + saveFileName + "\n" + help::stringFroJson(saveData));
#endif // _DEBUG
}

