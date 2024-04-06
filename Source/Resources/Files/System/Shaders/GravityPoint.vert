#version 330 core

layout (location = 0) in vec3 a_position;

uniform mat4 u_matProjectionView;

out vec3 position;

void main() {
	gl_Position = u_matProjectionView * vec4(a_position, 1.0);
	position = gl_Position.xyz;
}
