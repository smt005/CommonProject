#version 330 core
precision mediump float;

uniform vec4 u_color;
uniform float u_factor;
uniform float u_range;
uniform float u_rangeZ;

in vec3 positionPV;
in vec3 position;

void main() {
	vec4 color = u_color;
	
	float a = (u_range - length(positionPV)) / u_range;
	if (a < 0.0) {
		a = 0.0;
	} 
	else if (a > 1.0) {
		a = 1.0;
	}
	
	float aZ = (u_rangeZ + position.z) / u_rangeZ;
	if (aZ < 0.0) {
		aZ = 0.0;
	} 
	else if (aZ > 1.0) {
		aZ = 1.0;
	}
	
	gl_FragColor.a = a * aZ * u_factor;
}
