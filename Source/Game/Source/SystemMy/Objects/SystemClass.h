#pragma once

#define SYSTEM_MAP 7

#if SYSTEM_MAP == 0

namespace S00 {
	class SystemMap;
	class Body;
}
using SystemMap = S00::SystemMap;
using Body = S00::Body;

#elif SYSTEM_MAP == 1

namespace ARR {
	class SystemMap;
	class Body;
}
using SystemMap = ARR::SystemMap;
using Body = ARR::Body;

#elif SYSTEM_MAP == 3

namespace STATIC_ARR {
	class SystemMap;
	class Body;
	class SystemStackData;
}
using SystemMap = STATIC_ARR::SystemMap;
using Body = STATIC_ARR::Body;
using SystemStackData = STATIC_ARR::SystemStackData;

#elif SYSTEM_MAP == 4

namespace MY_VEC_ARR {
	class SystemMap;
	class Body;
	class SystemStackData;
}
using SystemMap = MY_VEC_ARR::SystemMap;
using Body = MY_VEC_ARR::Body;
using SystemStackData = MY_VEC_ARR::SystemStackData;

#elif SYSTEM_MAP == 5

namespace MAP_DOUBLE {
	class SystemMap;
	class Body;
}
using SystemMap = MAP_DOUBLE::SystemMap;
using Body = MAP_DOUBLE::Body;

#elif SYSTEM_MAP == 6

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

#endif
