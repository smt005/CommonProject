#pragma once

#include <vector>

class SpatialGrid final {
private:
	struct Point {
		float x, y, z;
		Point() = default;
		Point(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
	};

public:
	SpatialGrid();

	const float* const Data() const {
		return reinterpret_cast<const float*>(_points.data());
	}

	unsigned int Count() const {
		return _points.size();
	}

	unsigned short int Type() const {
		return 0x0003; // GL_LINE_STRIP 0x0003
	}

	const float* const const Color() const {
		return _color;
	}

	float WidthLine() const {
		return _widthLine;
	}

private:
	std::vector<Point> _points;
	float _widthLine = 1.f;
	float _color[4] = { 0.125f, 0.125f, 0.5f, 0.125f };
};
