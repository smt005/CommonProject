#version 330 core

layout (location = 0) in vec3 a_position;

uniform mat4 u_matProjectionView;
uniform float u_range;
uniform float u_mass_factor;

uniform vec3[100] u_splashPosition;
uniform float[100] u_distances;
uniform int u_splashCount;
	
uniform int u_body_count;
uniform vec3[100] u_body_positions;
uniform float[100] u_body_massess;
uniform vec3[100] u_body_colors;

out vec3 positionPV;
out vec3 position;
out vec3 oColor;

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
	oColor = vec3(0.0);
	
	position = a_position;
	float sumForce = 0.0;
	
	for (int i = 0; i < u_body_count; ++i) {
		float dist = distance(u_body_positions[i], position);
		float force = constGravity * u_body_massess[i] / (dist * dist);
		sumForce += force * u_mass_factor;
		
		vec3 oColorT = u_body_colors[i] * GetValue(u_range * 0.5, dist);
		oColorT.r = pow(oColorT.r, 15);
		oColorT.g = pow(oColorT.g, 15);
		oColorT.b = pow(oColorT.b, 15);
		oColor += oColorT;
	}
	
	for (int i = 0; i < u_splashCount; ++i) {
		float splashDist = distance(u_splashPosition[i], position);
		
		float k = 0.0;
		if (abs(splashDist - u_distances[i]) < 1.0) {
			k = 10.0;
		}
		else {
			k = abs(u_distances[i] / (u_distances[i] - splashDist)) / 5.0;
		}
		
		sumForce -= k;
	}
	
	position.z = sumForce;
	
	gl_Position = u_matProjectionView * vec4(position, 1.0);
	
	positionPV = gl_Position.xyz;
	gl_PointSize = 3.0 * GetValue(u_range * 0.5, length(positionPV));
}
