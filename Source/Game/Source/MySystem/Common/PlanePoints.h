// ◦ Xyz ◦

#include <vector>
#include <memory>
#include "Math/Vector.h"

//class Body;
//using BodyPtr = std::shared_ptr<Body>;
#include "../Objects/Body.h"
#include <glm/mat4x4.hpp>

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
};
