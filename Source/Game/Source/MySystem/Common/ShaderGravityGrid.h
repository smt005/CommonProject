#pragma once

#include <Draw2/Shader/ShaderInterface.h>
#include <MyStl/Singleton.h>

class ShaderGravityGrid final : public ShaderInterface, public mystd::Singleton<ShaderGravityGrid> {
public:
	void Use() override;
	void GetLocation() override;

public:
	static unsigned int u_matProjectionView;
	static unsigned int u_color;
	static unsigned int u_factor;
	static unsigned int u_range;
	static unsigned int u_rangeZ;

	static unsigned int u_body_count;
	static unsigned int u_body_positions;
	static unsigned int u_body_massess;
	static unsigned int u_body_colors;
};
