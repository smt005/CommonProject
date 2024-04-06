// ◦ Xyz ◦

#include <vector>
#include <memory>
#include "Math/Vector.h"
#include "../Objects/Body.h"

class PlanePoints final {
public:
	void Init(float space, float offset);
	void Update(std::vector<Body::Ptr>& objects);
	void Draw();

private:
	std::vector<Math::Vector3> _points;

	float _offset = 10.f;
	float _space = 1000.f;
	float _factor = 1.f;
	float _constGravity = -1.f;
	float _mass = 5.f;
};
