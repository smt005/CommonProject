#pragma once

#define SYSTEM_MAP 1

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
#endif