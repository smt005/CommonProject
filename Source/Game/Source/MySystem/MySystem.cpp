
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
#include <Object/Samples/Plane.h>
#include <Object/Samples/Box.h>
#include "ImGuiManager/UI.h"
#include "UI/MainUI.h"
#include "Math/Vector.h"

#include "Objects/BaseSpace.h"
#include "Objects/SpaceManager.h"
#include "Quests/Quest.h"
#include "Quests/QuestManager.h"
#include "Quests/Quests.h"
#include "Commands/Commands.h"
#include <Draw2/Draw2.h>
#include <Draw2/Shader/ShaderLine.h>
#include <Draw2/Shader/ShaderDefault.h>>

#include <../../CUDA/Source/Wrapper.h>
#include "Objects/SpaceTree02.h"

#define DRAW DrawLight
std::string MySystem::_resourcesDir;
std::shared_ptr<Space> MySystem::currentSpace;

const std::string saveFileName("../../../Executable/Save.json");

double minDt = std::numeric_limits<double>::max();
double maxDt = std::numeric_limits<double>::min();

MySystem::~MySystem() {
	if (_plane) { delete _plane; }
	if (_box) { delete _box; }
	if (_box2) { delete _box2; }
}

void MySystem::init() {
	CUDA::GetProperty();

	ShaderDefault::Instance().Init("Default.vert", "Default.frag");
	/*{
		float color4[] = { 1.f, 1.f, 1.0f, 0.5f };
		ShaderDefault::Instance().Use();
		Draw2::SetColorClass<ShaderDefault>(color4);
	}*/

	//ShaderLine::Instance().Init("Line.vert", "Line.frag");
	ShaderLineP::Instance().Init("LineP.vert", "Line.frag");
	//ShaderLinePM::Instance().Init("LinePM.vert", "Line.frag");

	CommandManager::Run("Commands/Main.json");
	//CommandManager::Run("Commands/EmptySpace.json");
	Init—ameras();
	initCallback();

	MainUI::Open(this);
}

void MySystem::close() {
	save();
	MainUI::Hide();
}

void MySystem::update() {
	QuestManager::Update();

	if (!currentSpace) {
		return;
	}

	if ((Engine::Core::currentTime() - _time) < (currentSpace->deltaTime > 33 ? 33 : currentSpace->deltaTime)) {
		return;
	} else {
		_time = Engine::Core::currentTime();
	}

	currentSpace->Update(1);

	auto[hasBody, body] = currentSpace->RefFocusBody();
	if (hasBody) {
		auto centerMassT = body.GetPos();
		glm::vec3 centerMass = glm::vec3(centerMassT.x, centerMassT.y, centerMassT.z);
		if (auto camera = dynamic_cast<CameraControlOutside*>(_camearCurrent.get())) {
			camera->SetPosOutside(centerMass);
		} else if (auto camera = dynamic_cast<Camera*>(_camearCurrent.get())) {
			centerMass.z = 100000.f;
			camera->SetPos(centerMass);
		}
	}
}

void MySystem::draw() {
	if (!currentSpace) {
		return;
	}

	Camera::Set<Camera>(_camearCurrent);

	Draw2::Viewport();
	Draw2::ClearColor();

	//ShaderDefault::current->Use();
	
	
	//Draw2::DepthTest(false);

	// SkyBox	
	/*if (MainUI::GetViewType() == 0 && currentSpace->_skyboxObject) {
		auto camPos = Camera::_currentCameraPtr->Pos();
		currentSpace->_skyboxObject->setPos(camPos);
		Draw2::SetModelMatrix(glm::mat4x4(1.f));

		Draw2::Draw(currentSpace->_skyboxObject->getModel());
		Draw2::ClearDepth();
	}*/

	//...
	if (CommonData::bool1) {
		ShaderDefault::Instance().Use();

		for (Body::Ptr& bodyPtr : currentSpace->_bodies) {
			if (!bodyPtr->visible) {
				continue;
			}

			Draw2::SetModelMatrixClass<ShaderDefault>(bodyPtr->getMatrix());
			Draw2::SetColorClass<ShaderDefault>(bodyPtr->getModel().getDataPtr());
			{
				float color4[] = { 1.f, 1.f, 1.0f, 1.0f };
				Draw2::SetColorClass<ShaderDefault>(color4);
			}
			Draw2::Draw(bodyPtr->getModel());
		}
	}

	//...
	if (CommonData::bool3) {
		if (SpaceTree02* space = dynamic_cast<SpaceTree02*>(currentSpace.get())) {
			Draw2::DepthTest(false);

			ShaderDefault::Instance().Use();

			Draw2::SetModelMatrix(glm::mat4x4(1.f));

			for (auto& clusterPtr : space->buffer) {
				//static float color4[] = { 0.f, 1.f, 0.0f, 0.1f };
				Draw2::SetColorClass<ShaderDefault>(clusterPtr->boxPtr->getDataPtr());
				Draw2::Draw(*clusterPtr->boxPtr);
			}

			ShaderLineP::Instance().Use();
			for (auto& clusterPtr : space->buffer) {
				//static float color4[] = { 0.f, 0.f, 1.0f, 1.f };
				Draw2::SetColorClass<ShaderLineP>(clusterPtr->boxPtr->getDataPtr());
				Draw2::Lines(*clusterPtr->boxPtr);
			}
		}

	}

	if (CommonData::bool4) {
		if (_plane) {
			Draw2::SetModelMatrix(glm::mat4x4(1.f));
			Draw2::Draw(*_plane);
		}
		else {
			//_plane = new Plane(Math::Vector3(-100.f, -100.f, 0.f), Math::Vector3(10.f, 10.f, 0.f));
		}

		//...
		if (_box && _box2) {
			ShaderDefault::Instance().Use();
			Draw2::SetModelMatrix(glm::mat4x4(1.f));

			{
				static float color4[] = { 0.f, 1.f, 0.f, 0.333f };
				Draw2::SetColorClass<ShaderDefault>(color4);
				Draw2::Draw(*_box);
			}
			{
				static float color4[] = { 0.f, 0.f, 1.f, 0.333f };
				Draw2::SetColorClass<ShaderDefault>(color4);
				Draw2::Draw(*_box2);
			}

			ShaderLineP::Instance().Use();
			{
				static float color4[] = { 0.f, 1.f, 0.f, 0.5f };
				Draw2::SetColorClass<ShaderLineP>(color4);
				Draw2::Lines(*_box);
			}
			{
				static float color4[] = { 0.f, 0.f, 1.f, 1.f };
				Draw2::SetColorClass<ShaderLineP>(color4);
				Draw2::Lines(*_box2);
			}

		}
		else {
			_box = new Box(Math::Vector3(-200.f, -100.f, -100.f), Math::Vector3(-10.f, -10.f, -10.f));
			_box2 = new Box(Math::Vector3(10.f, 10.f, 10.f), Math::Vector3(100.f, 200.f, 100.f));
		}
	}

	//...
	if (CommonData::bool2) {
		ShaderLineP::Instance().Use();

		static float sizePoint = 1.f;
		Draw2::SetPointSize(sizePoint);

		static float color4[] = { 1.f, 1.f, 0.5f, 1.f };
		Draw2::SetColorClass<ShaderLineP>(color4);

		unsigned int countPoints = currentSpace->_bodies.size();
		std::vector<float> points;
		points.reserve(countPoints * 3);

		for (Body::Ptr& bodyPtr : currentSpace->_bodies) {
			if (!bodyPtr->visible) {
				continue;
			}

			auto pos = bodyPtr->GetPos();

			points.emplace_back(pos.x);
			points.emplace_back(pos.y);
			points.emplace_back(pos.z);
		}

		Draw2::drawPoints(points.data(), countPoints);
	}

	//Draw2::DepthTest(true);
	//MainUI::DrawOnSpace();
}

void MySystem::draw2() {
	Camera::Set<Camera>(_camearCurrent);

	DRAW::viewport();
	DRAW::clearColor();
	DRAW::prepare();

	// SkyBox	
	if (MainUI::GetViewType() == 0 && currentSpace->_skyboxObject) {
		auto camPos = Camera::_currentCameraPtr->Pos();
		currentSpace->_skyboxObject->setPos(camPos);
		DRAW::draw(*currentSpace->_skyboxObject);
		DRAW::clearDepth();
	}

	// Grid
	Drawline();

	// Draw
	DRAW::prepare();
	DRAW::DrawMap(*currentSpace);
	
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

	if (currentSpace) {
		/*if (currentSpace->_skyboxModel) {
			DRAW::draw(*currentSpace->_skyboxModel);
		}*/

		auto [hasBody, star] = currentSpace->RefFocusBody();
		if (hasBody) {
			auto centerMassT = star.GetPos();
			glm::vec3 centerMass = glm::vec3(centerMassT.x, centerMassT.y, centerMassT.z);
			float cm[3] = { centerMass.x, centerMass.y, centerMass.z };
			DrawLine::SetIdentityMatrixByPos(cm);
			DrawLine::Draw(currentSpace->spatialGrid);
		}
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
				Math::Vector3 cursorPos(cursorPosGlm.x, cursorPosGlm.y, cursorPosGlm.z);
				SpaceManager::AddObjectOnOrbit(currentSpace.get(), cursorPos);
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
				dist -= Engine::Callback::pressKey(Engine::VirtualKey::SHIFT) ? 5000.f : 100.f;
			}
			else if (tapCallbackEvent->_id == Engine::VirtualTap::SCROLL_BOTTOM) {
				dist += Engine::Callback::pressKey(Engine::VirtualKey::SHIFT) ? 5000.f : 100.f;
			}

			if (dist < 10.f) {
				dist = 10.f;
			}
			cameraPtr->SetDistanceOutside(dist);
		}
	});
}

void MySystem::Init—ameras() {
	_camearSide = std::make_shared<CameraControlOutside>();
	if (CameraControlOutside* cameraPtr = dynamic_cast<CameraControlOutside*>(_camearSide.get())) {
		cameraPtr->SetPerspective(10000000.f, 1.f, 45.f);
		cameraPtr->SetPos({ 0, 0, 0 });
		cameraPtr->SetDirect({ -0.440075815f, -0.717726171f, -0.539631844f });
		cameraPtr->SetSpeed(1.0);
		cameraPtr->SetDistanceOutside(500.f);
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

	_camearCurrent = _camearSide;
	Camera::Set<Camera>(_camearCurrent);
	_camearScreen = std::make_shared<Camera>(Camera::Type::ORTHO);
}
