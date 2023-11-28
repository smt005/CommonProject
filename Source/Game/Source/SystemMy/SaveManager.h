#pragma once

class SaveManager final {
public:
	static void GetMap();
	static void Save();
	static bool Load();
	static void Reload();
};
