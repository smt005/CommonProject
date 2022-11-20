
#include "TouchGame/TouchGame.h"

#define NAME_GAME TouchGame

Engine::Game::Ptr Engine::Game::GetGame(const std::string& params) {
	Engine::Game::Ptr gamePtr(new NAME_GAME());
	return gamePtr;
}
