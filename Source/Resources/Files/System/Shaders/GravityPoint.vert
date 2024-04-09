#version 330 core

layout (location = 0) in vec3 a_position;

uniform mat4 u_matProjectionView;
uniform float u_range;

out vec3 positionPV;
out vec3 position;

float GetValue(float range, float dist)
{
	float a = (range - dist) / range;
	if (a < 0.0) {
		a = 0.0;
	} 
	else if (a > 1.0) {
		a = 1.0;
	}
	return a;
}

void main() {
	gl_Position = u_matProjectionView * vec4(a_position, 1.0);
	positionPV = gl_Position.xyz;
	position = a_position;
	gl_PointSize = 3.0 * GetValue(u_range * 0.5, length(positionPV));
}
