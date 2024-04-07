// ◦ Xyz ◦

#include <vector>
#include <memory>
#include <Object/Color.h>
#include "Math/Vector.h"
#include "../Objects/Body.h"


class PlanePoints final {
public:
	void Init(float spaceRange, float offset);
	void Update(std::vector<Body::Ptr>& objects);
	void Draw();

private:
	std::vector<Math::Vector3> _points;
	std::vector<Math::Vector3> _line;
	Math::Vector3 _bodyPos;
	Color _bodyColor;

	float _offset = 10.f;
	float _spaceRange = 1000.f;
	float _factor = 1.f;
	float _constGravity = -1.f;
	float _mass = 5.f;
};
