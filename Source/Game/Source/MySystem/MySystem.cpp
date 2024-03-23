// ◦ Xyz ◦
// ◦ Xyz ◦

#include "MySystem.h"
#include "Core.h"
#include <../../CUDA/Source/Wrapper.h>
#include "Callback/Callback.h"
#include "Callback/CallbackEvent.h"
#include "Common/Help.h"
#include "Draw/Camera/CameraControl.h"
#include "Draw/Camera/CameraControlOutside.h"
#include <Draw2/Draw2.h>
#include <Draw2/Shader/ShaderLine.h>
#include <Draw2/Shader/ShaderDefault.h>>
#include "ImGuiManager/UI.h"
#include "UI/MainUI.h"
#include "UI/CommonData.h"
#include "Object/Line.h"
#include "Objects/Space.h"
#include "Objects/SpaceTree02.h"
#include "Quests/QuestManager.h"
#include "Commands/Commands.h"

std::shared_ptr<Space> MySystem::currentSpace;
const std::string saveFileName("../../../Executable/Save.json");

void MySystem::init() {
	CUDA::GetProperty();
	ShaderDefault::Instance().Init("Default.vert", "Default.frag");
	ShaderLineP::Instance().Init("LineP.vert", "Line.frag");

	InitСameras();
	initCallback();
	
	MainUI::Open();
	CommandManager::Run("Commands/Main.json");
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

	// DEPRICATE_
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

	Draw2::Viewport();
	Draw2::ClearColor();
	Camera::Set<Camera>(_camearCurrent);

	// SkyBox
	if (CommonData::bool5) {
		if (!currentSpace->_skyboxModel) {
			currentSpace->_skyboxModel = Model::getByName("SkyBox");
		}

		if (MainUI::GetViewType() == 0 && currentSpace->_skyboxModel) {
			auto camPos = Camera::_currentCameraPtr->Pos();
			glm::mat4x4 mat(1.f);
			mat[3][0] = camPos.x;
			mat[3][1] = camPos.y;
			mat[3][2] = camPos.z;

			ShaderDefault::Instance().Use();
			Draw2::SetModelMatrix(mat);

			Draw2::Draw(*currentSpace->_skyboxModel);
			Draw2::ClearDepth();
		}
	}

	//...
	if (CommonData::bool1) {
		ShaderDefault::Instance().Use();
		Draw2::DepthTest(true);
		
		float color4[] = { 1.f, 1.f, 1.f, 1.f };
		Draw2::SetColorClass<ShaderDefault>(color4);
	
		for (Body::Ptr& bodyPtr : currentSpace->_bodies) {
			if (!bodyPtr->visible) {
				continue;
			}

			Draw2::SetModelMatrixClass<ShaderDefault>(bodyPtr->getMatrix());
			//Draw2::SetColorClass<ShaderDefault>(bodyPtr->getModel().getDataPtr());
			Draw2::Draw(bodyPtr->getModel());
		}
	}

	//...
	if (CommonData::bool3) {
		if (SpaceTree02* space = dynamic_cast<SpaceTree02*>(currentSpace.get())) {
			Draw2::DepthTest(false);

			// Кубы
			ShaderDefault::Instance().Use();
			Draw2::SetModelMatrix(glm::mat4x4(1.f));

			for (auto& clusterPtr : space->buffer) {
				Draw2::SetColorClass<ShaderDefault>(clusterPtr->boxPtr->getDataPtr());
				Draw2::Draw(*clusterPtr->boxPtr);
			}

			// Линии
			ShaderLineP::Instance().Use();
			Draw2::SetModelMatrix(glm::mat4x4(1.f));

			for (auto& clusterPtr : space->buffer) {
				Draw2::SetColorClass<ShaderLineP>(clusterPtr->boxPtr->getDataPtr());
				Draw2::Lines(*clusterPtr->boxPtr);
			}
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

	if (!CommonData::nameImageList.empty()) {
		Camera::Set<Camera>(_camearScreen);
		Draw2::DepthTest(false);

		ShaderDefault::Instance().Use();

		float color4[] = { 1.f, 1.f, 1.f, 1.f };
		Draw2::SetColorClass<ShaderDefault>(color4);
		Draw2::SetModelMatrix(glm::mat4x4(1.f));

		for (const std::string& nameModel : CommonData::nameImageList) {
			if (Model::Ptr& model = Model::getByName(nameModel)) {
				Draw2::Draw(*model);
			}
		}
	}

	//...
	if (CommonData::bool6) {
		if (Model::Ptr& model = Model::getByName("Finger")) {
			Camera::Set<Camera>(_camearScreen);
			Draw2::DepthTest(false);

			ShaderDefault::Instance().Use();

			float color4[] = { 1.f, 1.f, 1.f, 1.f };
			Draw2::SetColorClass<ShaderDefault>(color4);
			Draw2::SetModelMatrix(glm::mat4x4(1.f));

			Draw2::Draw(*model);
		}
	}

	//...
	if (CommonData::bool7) {
		if (Model::Ptr& model = Model::getByName("Logo")) {
			Camera::Set<Camera>(_camearScreen);
			Draw2::DepthTest(false);

			ShaderDefault::Instance().Use();

			float color4[] = { 1.f, 1.f, 1.f, 1.f };
			Draw2::SetColorClass<ShaderDefault>(color4);
			Draw2::SetModelMatrix(glm::mat4x4(1.f));

			Draw2::Draw(*model);
		}
	}

	// MainUI::DrawOnSpace();
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
	_callbackPtr = std::make_shared<Engine::Callback>(Engine::CallbackType::SCROLL, [this](const Engine::CallbackEventPtr& callbackEventPtr) {
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

void MySystem::InitСameras() {
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
