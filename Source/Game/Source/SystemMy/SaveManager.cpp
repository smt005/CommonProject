#include "SaveManager.h"
#include <memory>
#include <string>
#include <glm/vec3.hpp>

#include "../../Engine/Source/Common/Help.h"
#include "../../Engine/Source/Draw/Camera/CameraControlOutside.h"

namespace {
	std::string filePath = "Systems.json";
}

void SaveManager::GetMap() {
	Load();
}

void SaveManager::Save() {
	

}

bool SaveManager::Load() {
	return false;
}

void SaveManager::Reload() {
	GetMap();
}
