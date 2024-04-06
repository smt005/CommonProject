#version 330 core
precision mediump float;

uniform vec4 u_color;

in vec3 position;

void main() {
	float range = 1000.0;
	vec4 color = u_color;
	float a = (range - length(position)) / range;
	
	if (a < 0.0) {
		gl_FragColor.a = 0.0;
	} else {
		a *= 0.5;
		gl_FragColor.a = a;
	}
}
