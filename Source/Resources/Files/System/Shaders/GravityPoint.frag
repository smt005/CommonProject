#version 330 core
precision mediump float;

uniform vec4 u_color;
uniform vec3 u_body_position;
uniform vec4 u_body_color;
uniform float u_factor;
uniform float u_range;
uniform float u_rangeZ;

in vec3 positionPV;
in vec3 position;

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
	vec4 color = u_color * u_body_color * GetValue(50, length(position - u_body_position));
	color.a = GetValue(u_range, length(positionPV)) * GetValue(u_rangeZ, -position.z) * u_factor;
	
	gl_FragColor = color;
}
