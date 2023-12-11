#pragma once

#define SYSTEM_MAP 8

#if SYSTEM_MAP == 6

namespace MAP_EASY_MERGE {
	class SystemMap;
	class Body;
}
using SystemMap = MAP_EASY_MERGE::SystemMap;
using Body = MAP_EASY_MERGE::Body;

#elif SYSTEM_MAP == 7

class SystemMap;
class Body;

using SystemMap = SystemMap;
using Body = Body;

#elif SYSTEM_MAP == 8

class SystemMap;
class Body;

using SystemMap = SystemMap;
using Body = Body;

#endif
