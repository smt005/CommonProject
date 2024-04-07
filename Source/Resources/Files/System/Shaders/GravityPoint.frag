#version 330 core
precision mediump float;

uniform vec4 u_color;
uniform float u_factor;
uniform float u_range;
uniform float u_rangeZ;

in vec3 positionPV;
in float positionZ;

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
	vec4 color = u_color;
	color.a = GetValue(u_range, length(positionPV)) * GetValue(u_rangeZ, -positionZ) * u_factor;
	gl_FragColor = color;
}
