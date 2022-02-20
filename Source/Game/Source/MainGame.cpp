
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

#include <memory>

std::string MainGame::_resourcesDir;

MainGame::MainGame()
	: _indexCurrentMap(-1)
	, _updateTime (0)
{
}

MainGame::~MainGame() {
	_callbackPtr.reset();
}

void MainGame::init() {
	// Camera
	Camera::current.setFromEye(true);
	Camera::current.setPos(glm::vec3(0.f, 95.f, 0.f));
	Camera::current.setVector(glm::vec3(0.f, -1.f, 0.f));
	Camera::current.setDist(1.0f);

	Draw::setClearColor(0.9f, 0.6f, 0.3f, 1.0f);

	if (_indexCurrentMap == -1) {
		_maps.emplace_back("Table");
		_indexCurrentMap = 0;
	}

	_visibleMap["Table_01"] = false;
	_visibleMap["Table_02"] = false;
	_visibleMap["Table_03"] = false;

	update();

	initCallback();
}

void MainGame::update() {
	double currentTime = Engine::Core::currentTime();
	if (_updateTime > currentTime) {
		return;
	}

	_updateTime = currentTime + 3000;

	// Смена видимости и ...
	for (auto& it = _visibleMap.begin(); it != _visibleMap.end(); ++it) {
		float value = help::random(0.f, 1000.f);
		it->second = value > 500.f ? true : false;
	}

	// Смена отображения
	if (Map::Ptr map = Map::getByName("Table")) {
		for (const std::pair<std::string, bool>& pair : _visibleMap) {		
			if (Object* object = map->getObjectByName(pair.first)) {
				object->setVisible(pair.second);
			}
		}
	}
}

void MainGame::draw() {
	Draw::viewport();
	Draw::clearColor();

	// Draw
	Draw::prepare();
	Draw::draw(*Map::getByName(_maps[_indexCurrentMap]));
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

		}
	});

	_callbackPtr->add(Engine::CallbackType::RELEASE_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (key == Engine::VirtualKey::ESCAPE) {
			Engine::Core::close();
		}

		if (key == Engine::VirtualKey::A) {
			changeMap(false);
		}

		if (key == Engine::VirtualKey::D) {
			changeMap(true);
		}

		std::string nameObject;
		if (key == Engine::VirtualKey::VK_1) { nameObject = "Table_01"; }
		if (key == Engine::VirtualKey::VK_2) { nameObject = "Table_02"; }
		if (key == Engine::VirtualKey::VK_3) { nameObject = "Table_03"; }

		if (!nameObject.empty()) {
			if (Map::Ptr map = Map::getByName("Table")) {
				if (Object* object = map->getObjectByName(nameObject)) {
					bool visible = !object->visible();
					object->setVisible(visible);
				}
			}
		}

		if (key == Engine::VirtualKey::Q) {
			if (Map::Ptr map = Map::getByName("Table")) {
				if (Object* object = map->getObjectByName("Table_00")) {
					bool visible = !object->visible();
					object->setVisible(visible);
				}
				if (Object* object = map->getObjectByName("Table_99")) {
					bool visible = !object->visible();
					object->setVisible(visible);
				}
			}
		}

		if (key == Engine::VirtualKey::R && Engine::Callback::pressKey(Engine::VirtualKey::CONTROL)) {
			Model::clear();
			Model::removeData();
			Texture::clear();
			Shape::clear();
			Map::getByName(_maps[_indexCurrentMap])->load();
		}
	});
}

void MainGame::initPhysic() {

}

bool MainGame::load() {

	return false;
}

void MainGame::save() {

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
