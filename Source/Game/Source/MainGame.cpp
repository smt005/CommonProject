
#include "MainGame.h"
#include "Core.h"
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"
#include "Callback/CallbackEvent.h"
#include "Screen.h"
#include "Common/Help.h"
#include "Draw/Camera.h"
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
#include "Windows/EditMap.h"
#include "Windows/Console.h"

#include <memory>
#include <set>

#define DRAW DrawLight

std::string MainGame::_resourcesDir;
const std::string saveFileName("../../../Executable/Save.json");
Object lightObject;
vec3 lightPos(-500.0f, -500.0f, 500.0f);

MainGame::MainGame()
	: _indexCurrentMap(-1)
	, _updateTime (0)
	, _state(State::MENU)
{
}

MainGame::~MainGame() {
	_callbackPtr.reset();

	if (_greed) {
		delete _greed;
		_greed = nullptr;
	}
}

void MainGame::init() {
	_greed = new Greed(100.0f, 10.0f);
	_greed->setPos({ 0.0f, 0.0f, 0.1f });

	if (!load()) {
		Camera::current.setFromEye(true);
		Camera::current.setPos(glm::vec3(0.f, 95.f, 0.f));
		Camera::current.setVector(glm::vec3(0.f, -1.f, 0.f));
		Camera::current.setDist(1.0f);
	}

	DRAW::setClearColor(0.3f, 0.6f, 0.9f, 1.0f);

	if (_indexCurrentMap == -1) {
		{
			std::map<std::string, bool> visibleMap;
			_maps.emplace_back(std::pair("Apartment_00", visibleMap));
		}
		{
			std::map<std::string, bool> visibleMap;
			_maps.emplace_back(std::pair("Apartment_01", visibleMap));
		}
		/* {
			std::map<std::string, bool> visibleMap;
			_maps.emplace_back(std::pair("Apartment_02", visibleMap));
		}
		{
			std::map<std::string, bool> visibleMap;
			_maps.emplace_back(std::pair("Apartment_03", visibleMap));
		}*/

		/* {
			std::map<std::string, bool> visibleMap;
			_maps.emplace_back(std::pair("Room_00", visibleMap));
		}
		{
			std::map<std::string, bool> visibleMap;
			_maps.emplace_back(std::pair("Room_01", visibleMap));
		}
		{
			std::map<std::string, bool> visibleMap;
			_maps.emplace_back(std::pair("Room_02", visibleMap));
		}*/

		_indexCurrentMap = 0;
		_state = MainGame::State::GAME;
	}

	//lightObject.set("light", "Lamp_00", lightPos);

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
	DRAW::viewport();
	DRAW::clearColor();

	// Draw
	DRAW::prepare();

	if (_state == MainGame::State::MENU) {
		DRAW::draw(*Map::getByName("Menu"));
	}
	else if (_state == MainGame::State::EXIT) {
		Engine::Core::close();
	}
	else {
		DRAW::draw(*Map::getByName(_maps[_indexCurrentMap].first));
	}

	// Свет
	Draw::prepare();
	Draw::draw(lightObject);
}

void MainGame::Drawline() {
	DrawLine::prepare();

	if (_qwe0_) {
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

		/* {
			float pointsX[] = { 0.f, 0.f, 0.f, _lenghtNormal, _lenghtNormal, _lenghtNormal };
			Line lineX(pointsX, 2, Line::LINE);
			lineX.setLineWidth(5.0f);
			lineX.color = Color::RED;
			lineX.color.setAlpha(0.999);
			DrawLine::draw(lineX);
		}*/
	}

	DrawLine::draw(*_greed);

	//...
	if (_qwe_) {
		Map& map = *Map::getByName(_maps[_indexCurrentMap].first);

		for (auto& object : map.objects) {
			if (object->visible()) {
				float* vertexes = object->getModel().getMesh().vertexes();
				float* normals = object->getModel().getMesh().normals();
				int countVertex = object->getModel().getMesh().countVertex();

				for (int i = 0; i < countVertex * 3; i = i + 3) {
					float* pos = &vertexes[i];
					float* norm = &normals[i];

					float pointsX[] = { pos[0], pos[1], pos[2], pos[0] + norm[0] * _lenghtNormal, pos[1] + norm[1] * _lenghtNormal, pos[2] + norm[2] * _lenghtNormal};
					Line lineX(pointsX, 2, Line::LINE);
					lineX.setLineWidth(_widthNormal);
					lineX.color = Color::RED;
					lineX.color.setAlpha(0.75);
					DrawLine::draw(lineX, object->getMatrix());
				}
			}
		}
	}
}

void MainGame::resize() {
	Camera::getCurrent().setPerspective(Camera::getCurrent().fov(), Engine::Screen::aspect(), 0.1f, 1000.0f);
}

void MainGame::initCallback() {
	_callbackPtr = std::make_shared<Engine::Callback>(Engine::CallbackType::PINCH_TAP, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Editor::Console::IsLock()) {
			return;
		}

		if (Engine::Callback::pressTap(Engine::VirtualTap::RIGHT)) {
			Camera::current.rotate(Engine::Callback::deltaMousePos());
			CheckMouse();
		}

		if (Engine::Callback::pressTap(Engine::VirtualTap::MIDDLE)) {
			Camera::current.move(Engine::Callback::deltaMousePos() * Engine::Core::deltaTime());
			CheckMouse();
		}

		if (Engine::Callback::pressTap(Engine::VirtualTap::LEFT)) {
			hit(Engine::Callback::mousePos().x, Engine::Screen::height() - Engine::Callback::mousePos().y, true);
		}
		});

	_callbackPtr->add(Engine::CallbackType::SCROLL, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Engine::TapCallbackEvent* tapCallbackEvent = dynamic_cast<Engine::TapCallbackEvent*>(callbackEventPtr.get())) {
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

	_callbackPtr->add(Engine::CallbackType::RELEASE_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Editor::Console::IsLock()) {
			return;
		}

		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (key == Engine::VirtualKey::ESCAPE) {
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

		if (key == Engine::VirtualKey::T) {
			DrawLight::resetShader();
		}
		});

	_callbackPtr->add(Engine::CallbackType::PINCH_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Editor::Console::IsLock()) {
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

	gml:vec3 posCam = Camera::current.pos();
		if (posCam.z < 50.f) {
			posCam.z = 50.f;
			Camera::current.setPos(posCam);
		}
		if (posCam.z > 270.f) {
			posCam.z = 270.f;
			Camera::current.setPos(posCam);
		}
		if (posCam.x < -320.f) {
			posCam.x = -320.f;
			Camera::current.setPos(posCam);
		}
		if (posCam.x > 320.f) {
			posCam.x = 320.f;
			Camera::current.setPos(posCam);
		}
		if (posCam.y < -235.f) {
			posCam.y = -235.f;
			Camera::current.setPos(posCam);
		}
		if (posCam.y > 375.f) {
			posCam.y = 375.f;
			Camera::current.setPos(posCam);
		}

		// Свет
		if (Engine::Callback::pressKey(Engine::VirtualKey::O)) {
			lightPos.z += 1.0f;
		}
		if (Engine::Callback::pressKey(Engine::VirtualKey::L)) {
			lightPos.z -= 1.0f;
		}
		if (Engine::Callback::pressKey(Engine::VirtualKey::H)) {
			lightPos.x += 1.0f;
		}
		if (Engine::Callback::pressKey(Engine::VirtualKey::K)) {
			lightPos.x -= 1.0f;
		}
		if (Engine::Callback::pressKey(Engine::VirtualKey::U)) {
			lightPos.y += 1.0f;
		}
		if (Engine::Callback::pressKey(Engine::VirtualKey::J)) {
			lightPos.y -= 1.0f;
		}

		lightObject.setPos(lightPos);
		DRAW::setLightPos(lightPos.x, lightPos.y, lightPos.z);
		});

	_callbackPtr->add(Engine::CallbackType::MOVE, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		if (Editor::Console::IsLock()) {
			return;
		}

		if (_state == State::MENU) {
			float currentMousePos[] = { Engine::Callback::mousePos().x, Engine::Screen::height() - Engine::Callback::mousePos().y };
			if (_mousePos[0] != currentMousePos[0] && _mousePos[1] != currentMousePos[1]) {
				_mousePos[0] = currentMousePos[0];
				_mousePos[1] = currentMousePos[1];
				hit(_mousePos[0], _mousePos[1]);
			}
		}
		});


	_callbackPtr->add(Engine::CallbackType::RELEASE_KEY, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
		Engine::KeyCallbackEvent* releaseKeyEvent = (Engine::KeyCallbackEvent*)callbackEventPtr->get();
		Engine::VirtualKey key = releaseKeyEvent->getId();

		if (key == Engine::VirtualKey::TILDE) {
			if (UI::ShowingWindow("Console")) {
				UI::CloseWindow("Console");
			}
			else {
				_editMapWindow = UI::ShowWindow<Editor::Console>();
			}
		}

		if (key == Engine::VirtualKey::F1) {
			if (UI::ShowingWindow("Edit map")) {
				UI::CloseWindow("Edit map");
			}
			else {
				_editMapWindow = UI::ShowWindow<Editor::Map>();
			}
		}
	});
}

void MainGame::CheckMouse() {
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
