#version 330 core

layout (location = 0) in vec3 a_position;

uniform mat4 u_matProjectionView;
uniform float u_range;

uniform int u_body_count;
uniform vec3[100] u_body_positions;
uniform float[100] u_body_massess;

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
	float constGravity = -0.2;
	
	position = a_position;
	float force = 0.0;
	
	for (int i = 0; i < u_body_count; ++i) {
		float dist = distance(u_body_positions[i], position);
		force += constGravity * u_body_massess[i] / (dist * dist);
	}
	
	position.z = force;
	
	gl_Position = u_matProjectionView * vec4(position, 1.0);
	positionPV = gl_Position.xyz;
	gl_PointSize = 3.0 * GetValue(u_range * 0.5, length(positionPV));
}
