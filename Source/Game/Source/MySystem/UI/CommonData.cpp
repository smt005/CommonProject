// ◦ Xyz ◦
#include "CommonData.h"

int CommonData::lockScreen = false;
int CommonData::lockAction = false;

bool CommonData::bool0 = false;
bool CommonData::bool1 = true;
bool CommonData::bool2 = false;
bool CommonData::bool3 = false;
bool CommonData::bool4 = false;
bool CommonData::bool5 = false;
bool CommonData::bool6 = false;
bool CommonData::bool7 = false;
bool CommonData::bool8 = false;
bool CommonData::bool9 = false;

std::string CommonData::textOnScreen;
std::vector<std::string> CommonData::nameImageList;

float* CommonData::Color4()
{
	static float color4[4] = { 1.f, 1.f, 1.f, 1.f };
	return color4;
}

// Lock screen

void CommonData::PushLockScreen()
{
	++lockScreenCounter;
}
void CommonData::PopLockScreen()
{
	--lockScreenCounter;
}
void CommonData::UnlockScreen()
{
	lockScreenCounter = 0;
}
bool CommonData::IsLockScreen()
{
	return lockScreenCounter > 0;
}
int CommonData::lockScreenCounter = 0;
