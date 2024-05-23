// ◦ Xyz ◦

#include <vector>
#include <memory>
#include <Object/Color.h>
#include "Math/Vector.h"
#include "../Objects/Body.h"


class GravityGrid final {
public:
	void Init(float spaceRange, float offset);
	void Draw();
	void Update(double dt);
	void AddTime(const Math::Vector3& pos, float dist);

public:
	static GravityGrid* gravityGridPtr;

private:
	std::vector<float> _distances;
	std::vector<Math::Vector3> _splashPosition;
	int _splashCount = 0;

	std::vector<Math::Vector3> _points;
	std::vector<Math::Vector3> _line;

	float _offset = 0.5f;
	float _spaceRange = 100.f;
	float _factor = 1.f;
	float _constGravity = -0.2f;
	float _mass = 10000.f;
};
