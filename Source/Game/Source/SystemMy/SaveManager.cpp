#include "SaveManager.h"
#include <memory>
#include <string>
#include <glm/vec3.hpp>

#include  "Objects/SystemMap.h"
#include "../../Engine/Source/Common/Help.h"
#include "../../Engine/Source/Object/Model.h"
#include "../../Engine/Source/Draw/Camera/CameraControlOutside.h"
namespace {
	std::string filePath = "Systems.json";
}

void SaveManager::GetMap() {
	Load();

	/*if (!(Body::system = Map::GetFirstCurrentMapPtr())) {
		BodyMy::system = Map::AddCurrentMap(Map::Ptr(new Map("GenerateMap")));
	}

	BodyMy* sun = nullptr;
	if (Object::Ptr objectPtr = BodyMy::system->getObjectPtrByName("Sun")) {
		sun = dynamic_cast<BodyMy*>(objectPtr.get());
	}

	if (!sun) {
		Object::Ptr sunPtr = std::make_shared<BodyMy>("Sun", "OrangeStar", glm::vec3(0.f, 0.f, 0.f));
		sun = static_cast<BodyMy*>(sunPtr.get());
		sun->setMass(1000000.f);
		BodyMy::system->addObject(sunPtr);
	}

	BodyMy::_suns.emplace_back(sun);*/
}

void SaveManager::Save() {
	

}

bool SaveManager::Load() {
	return false;
}

void SaveManager::Reload() {
	GetMap();
}
