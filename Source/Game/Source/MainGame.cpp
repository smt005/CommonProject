
#include "MainGame.h"
#include "Core.h"
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"
#include "Callback/CallbackEvent.h"
#include "Screen.h"
#include "Common/Help.h"
#include "Draw/Camera.h"
#include "Draw/Draw.h"
#include "Object/Map.h"
#include "Object/Object.h"
#include "Object/Model.h"
#include "Object/Shape.h"

#include "glm/vec2.hpp"

#include <memory>
#include <set>

std::string MainGame::_resourcesDir;
const std::string saveFileName("../../../Executable/Save.json");

MainGame::MainGame()
	: _indexCurrentMap(-1)
	, _updateTime (0)
	, _state(State::MENU)
{
}

MainGame::~MainGame() {
	_callbackPtr.reset();
}

void MainGame::init() {
	if (!load()) {
		Camera::current.setFromEye(true);
		Camera::current.setPos(glm::vec3(0.f, 95.f, 0.f));
		Camera::current.setVector(glm::vec3(0.f, -1.f, 0.f));
		Camera::current.setDist(1.0f);
	}

	Draw::setClearColor(0.9f, 0.6f, 0.3f, 1.0f);

	if (_indexCurrentMap == -1) {
		{
			std::map<std::string, bool> visibleMap;
			_maps.emplace_back(std::pair("Room_00", visibleMap));
		}
		{
			std::map<std::string, bool> visibleMap;
			_maps.emplace_back(std::pair("Room_01", visibleMap));
		}
		_indexCurrentMap = 0;
	}

	update();
	initCallback();
}

void MainGame::update() {
	if (_state == State::MENU) {
		return;
	} else if (_state == State::EXIT) {
		Engine::Core::close();
		return;
	}

	/*double currentTime = Engine::Core::currentTime();
	if (_updateTime > currentTime) {
		return;
	}

	_updateTime = currentTime + 3000;

	std::map<std::string, bool>& visibleMap = _maps[_indexCurrentMap].second;*/
}

void MainGame::draw() {
	Draw::viewport();
	Draw::clearColor();

	// Draw
	Draw::prepare();

	if (_state == MainGame::State::MENU) {
		Draw::draw(*Map::getByName("Menu"));
	}
	else if (_state == MainGame::State::EXIT) {
		Engine::Core::close();
	}
	else {
		Draw::draw(*Map::getByName(_maps[_indexCurrentMap].first));
	}
}

void MainGame::resize() {
	Camera::getCurrent().setPerspective(45.0f, Engine::Screen::aspect(), 0.1f, 1000.0f);
}

void MainGame::initCallback() {
	_callbackPtr = std::make_shared<Engine::Callback>(Engine::CallbackType::PINCH_TAP, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::Callback::pressTap(Engine::VirtualTap::RIGHT)) {
			Camera::current.rotate(Engine::Callback::deltaMousePos());
		}

		if (Engine::Callback::pressTap(Engine::VirtualTap::MIDDLE)) {
			Camera::current.move(Engine::Callback::deltaMousePos() * 1000.0f * Engine::Core::deltaTime());
		}

		if (Engine::Callback::pressTap(Engine::VirtualTap::LEFT)) {
			hit(Engine::Callback::mousePos().x, Engine::Screen::height() - Engine::Callback::mousePos().y, true);
		}
	});

	_callbackPtr->add(Engine::CallbackType::RELEASE_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (key == Engine::VirtualKey::ESCAPE) {
			//Engine::Core::close();
			_state = State::MENU;
			Map& map = currentMap();
			map.load();
			Camera::setCurrent(map.getCamera());
		}

		if (key == Engine::VirtualKey::S && Engine::Callback::pressKey(Engine::VirtualKey::CONTROL)) {
			save();
		}

		if (key == Engine::VirtualKey::L && Engine::Callback::pressKey(Engine::VirtualKey::CONTROL)) {
			load();
		}

		if (key == Engine::VirtualKey::Q) {
			changeMap(false);
			Map& map = currentMap();
			Camera::setCurrent(map.getCamera());
		}

		if (key == Engine::VirtualKey::E) {
			changeMap(true);
			Map& map = currentMap();
			Camera::setCurrent(map.getCamera());
		}

		if (key == Engine::VirtualKey::R && Engine::Callback::pressKey(Engine::VirtualKey::CONTROL) && Engine::Callback::pressKey(Engine::VirtualKey::SHIFT)) {
			Model::clear();
			Model::removeData();
			Texture::clear();
			Shape::clear();

			Map& map = currentMap();
			map.load();
			Camera::setCurrent(map.getCamera());
		}
		else if (key == Engine::VirtualKey::R && Engine::Callback::pressKey(Engine::VirtualKey::CONTROL)) {
			Map& map = currentMap();
			map.load();
			Camera::setCurrent(map.getCamera());
		}

	});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::Callback::pressKey(Engine::VirtualKey::CONTROL)) {
			return;
		}

		float speedCamera = 5.0f * Engine::Core::deltaTime();
		if (Engine::Callback::pressKey(Engine::VirtualKey::SHIFT)) {
			speedCamera = 30.0f * Engine::Core::deltaTime();
		}

		if (Engine::Callback::pressKey(Engine::VirtualKey::S)) {
			Camera::current.move(CAMERA_FORVARD, speedCamera);
		}

		if (Engine::Callback::pressKey(Engine::VirtualKey::W)) {
			Camera::current.move(CAMERA_BACK, speedCamera);
		}

		if (Engine::Callback::pressKey(Engine::VirtualKey::D)) {
			Camera::current.move(CAMERA_RIGHT, speedCamera);
		}

		if (Engine::Callback::pressKey(Engine::VirtualKey::A)) {
			Camera::current.move(CAMERA_LEFT, speedCamera);
		}

		if (Engine::Callback::pressKey(Engine::VirtualKey::R)) {
			Camera::current.move(CAMERA_TOP, speedCamera);
		}

		if (Engine::Callback::pressKey(Engine::VirtualKey::F)) {
			Camera::current.move(CAMERA_DOWN, speedCamera);
		}
	});

	_callbackPtr->add(Engine::CallbackType::MOVE, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		static glm::vec2 mousePos;
		glm::vec2 currentMousePos(Engine::Callback::mousePos().x, Engine::Screen::height() - Engine::Callback::mousePos().y);

		if (mousePos != currentMousePos) {
			hit(currentMousePos.x, currentMousePos.y);
			mousePos = currentMousePos;
		}
	});
}

void MainGame::initPhysic() {

}

bool MainGame::load()
{
	Json::Value saveData;
	if (!help::loadJson(saveFileName, saveData) || saveData.empty()) {
		return false;
	}

	if (!saveData["camera"].empty()) {
		Json::Value& cameraData = saveData["camera"];
		Camera::current.setJsonData(cameraData);
	}

#if _DEBUG
	Engine::Core::log("LOAD: " + saveFileName + "\n" + help::stringFroJson(saveData));
#endif // _DEBUG

	return true;
}

void MainGame::save()
{
	Json::Value saveData;

	Json::Value cameraData;
	Camera::current.getJsonData(cameraData);

	saveData["camera"] = cameraData;
	saveData["testKey"] = "testValue";

	help::saveJson(saveFileName, saveData);

#if _DEBUG
	Engine::Core::log("SAVE: " + saveFileName + "\n" + help::stringFroJson(saveData));
#endif // _DEBUG
}

Map& MainGame::currentMap() {
	std::string currentmapStr;
	if (_state == MainGame::State::MENU) {
		currentmapStr = "Menu";
	}
	else {
		currentmapStr = _maps[_indexCurrentMap].first;
	}

	if (currentmapStr.empty()) {
		currentmapStr = "Menu";
	}

	Map::Ptr mapPtr = Map::getByName(currentmapStr);
	return *mapPtr;
}

void MainGame::changeMap(const bool right) {
	size_t size = _maps.size();
	if (size < 2) {
		return;
	}

	_indexCurrentMap += right ? 1 : -1;

	if (_indexCurrentMap < 0) {
		_indexCurrentMap = size - 1;
	}
	else if (_indexCurrentMap >= size) {
		_indexCurrentMap = 0;
	}
}

void MainGame::hit(const int x, const int y, const bool action) {
	std::set<std::string> objectsUnderMouse;

	for (Object* object : currentMap().objects) {
		if (object->visible() && object->hit(x, y)) {
			objectsUnderMouse.emplace(object->getName());
		}
	}

	if (objectsUnderMouse.find("Door_00") != objectsUnderMouse.end() || objectsUnderMouse.find("Door_rotate_00") != objectsUnderMouse.end()) {
		currentMap().getObjectByName("Door_00").setVisible(false);
		currentMap().getObjectByName("Door_rotate_00").setVisible(true);
		
		if (objectsUnderMouse.find("Door_rotate_00") != objectsUnderMouse.end() && action) {
			_state = State::EXIT;
		}
	}
	else if (objectsUnderMouse.find("Monitor_display_00") != objectsUnderMouse.end() || objectsUnderMouse.find("Monitor_display_01") != objectsUnderMouse.end()) {
		currentMap().getObjectByName("Monitor_display_00").setVisible(false);
		currentMap().getObjectByName("Monitor_display_01").setVisible(true);

		if (objectsUnderMouse.find("Monitor_display_01") != objectsUnderMouse.end() && action) {
			_state = State::GAME;
			Map& map = currentMap();
			map.load();
			Camera::setCurrent(map.getCamera());
		}
	}
	else {
		currentMap().getObjectByName("Door_00").setVisible(true);
		currentMap().getObjectByName("Door_rotate_00").setVisible(false);
		currentMap().getObjectByName("Monitor_display_00").setVisible(true);
		currentMap().getObjectByName("Monitor_display_01").setVisible(false);
	}
}
