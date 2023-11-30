
#include "SystemMy/SystemMy.h"
#define NAME_GAME SystemMy

Engine::Game::Uptr Engine::Game::GetGame(const std::string& params) {
	Engine::Game::Uptr gameUptr(new NAME_GAME());
	return gameUptr;
}
