#include "Body.h"

#include "Object/Model.h"
#include "Common/Help.h"
#include "Callback/Callback.h"
#include "Draw/Camera/Camera.h"

Body::Body(const std::string& nameModel)
	: _model(Model::getByName(nameModel))
{}

Body::Body(const std::string& nameModel, const Math::Vector3d& pos, const Math::Vector3d& velocity, double mass, const std::string& name)
	: _mass(mass)
	, _velocity(velocity)
	, _model(Model::getByName(nameModel))
{
	SetPos(pos);
}

Body::~Body() {
	delete _name;
}
// TODO:
Math::Vector3 Body::PosOnScreen(const glm::mat4x4& matCamera, bool applySizeScreen) {
	Math::Vector3d posOnScreen;

	auto transformToScreen = [](Math::Vector3& point, const glm::mat4x4& mat) {
		glm::vec4 p(point.x, point.y, point.z, 1.0f);
		p = mat * p;

		if (p.w != 1.0)
		{
			p.x /= p.w;
			p.y /= p.w;
			p.z /= p.w;
		}

		point.x = p.x;
		point.y = p.y;
		point.z = p.z;
	};

	auto transformSizeToScreen = [applySizeScreen](float& x, float& y, float& xRes, float& yRes) {
		x = x * 0.5f + 0.5f;
		y = y * 0.5f + 0.5f;

		if (applySizeScreen) {
			xRes = Engine::Screen::width() * x;
			yRes = Engine::Screen::height() * y;
		}
	};

	Math::Vector3 point = GetPos();
	transformToScreen(point, matCamera);

	float xInt = point.x;
	float yInt = point.y;

	if (applySizeScreen) {
		transformSizeToScreen(point.x, point.y, xInt, yInt);
	}

	float aspectCScreen = Engine::Screen::aspect();
	if (aspectCScreen > 1.f) {
		posOnScreen.x = xInt * aspectCScreen;
		posOnScreen.y = yInt;
	} else {
		posOnScreen.x = xInt;
		posOnScreen.y = yInt / aspectCScreen;
	}
	posOnScreen.z = 0;

	return posOnScreen;
}

// TODO:
bool Body::hit(const glm::mat4x4& matCamera) {
	if (!_model) {
		return false;
	}

	const int xTap = Engine::Callback::mousePos().x;
	const int yTap = Engine::Screen::height() - Engine::Callback::mousePos().y;

	auto transformToScreen = [](glm::vec3& point, const glm::mat4x4& mat) {
		glm::vec4 p(point.x, point.y, point.z, 1.0f);
		p = mat * p;

		if (p.w != 1.0)
		{
			p.x /= p.w;
			p.y /= p.w;
			p.z /= p.w;
		}

		point.x = p.x;
		point.y = p.y;
		point.z = p.z;
	};

	auto transformSizeToScreen = [](float& x, float& y, int& xRes, int& yRes) {
		x = x * 0.5f + 0.5f;
		y = y * 0.5f + 0.5f;

		xRes = Engine::Screen::width() * x;
		yRes = Engine::Screen::height() * y;
	};

	auto hintTriangle = [](float* v0, float* v1, float* v2, int xTap, int yTap) {
		//координаты вершин треугольника
		float x1 = v0[0], y1 = v0[1];
		float x2 = v1[0], y2 = v1[1];
		float x3 = v2[0], y3 = v2[1];

		//координаты произвольной точки
		float x = (float)xTap, y = (float)yTap;

		float a = (x1 - x) * (y2 - y1) - (x2 - x1) * (y1 - y);
		float b = (x2 - x) * (y3 - y2) - (x3 - x2) * (y2 - y);
		float c = (x3 - x) * (y1 - y3) - (x1 - x3) * (y3 - y);
		bool res = ((a >= 0 && b >= 0 && c >= 0) || (a <= 0 && b <= 0 && c <= 0)) ? true : false;
		return res;
	};

	//glm::mat4x4 matCamera = Camera::GetLink().ProjectView();

	const Mesh& mesh = _model->getMesh();

	for (int index = 0; index < mesh.countIndex(); index += 3)
	{
		float vertexScreen[3][2];

		for (int iShift = 0; iShift < 3; ++iShift) {
			int a = mesh.indexes()[index + iShift];
			int index = 3 * a;
			glm::vec4 point4(mesh.vertexes()[index], mesh.vertexes()[index + 1], mesh.vertexes()[index + 2], 1.0f);
			point4 = _matrix * point4;

			glm::vec3 point(point4.x, point4.y, point4.z);
			transformToScreen(point, matCamera);

			int xInt, yInt;
			transformSizeToScreen(point.x, point.y, xInt, yInt);

			vertexScreen[iShift][0] = xInt;
			vertexScreen[iShift][1] = yInt;
		}

		if (hintTriangle(vertexScreen[0], vertexScreen[1], vertexScreen[2], xTap, yTap)) {
			return true;
		}
	}

	return false;
}

void Body::Rotate() {
	_angular += _angularVelocity;
	_matrix = glm::rotate(_matrix, _angular, { 0.f, 0.f, 1.f });
}

void Body::Scale() {
	constexpr float val3div4 = (3.f / 4.f) / PI;

	//  std::pow(n, 1/3.) (or std::cbrtf(v);
	_scale = std::cbrtf(val3div4 * _mass);

	float x = _matrix[3][0];
	float y = _matrix[3][1];
	float z = _matrix[3][2];

	_matrix = glm::scale(glm::mat4x4(1.f), glm::vec3(_scale));

	_matrix[3][0] = x;
	_matrix[3][1] = y;
	_matrix[3][2] = z;
}