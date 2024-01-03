#version 330 core

uniform mat4 u_matProjectionView;
uniform mat4 u_matViewModel;

layout (location = 0) in vec4 a_position;

void main() {
	gl_Position = u_matProjectionView * u_matViewModel * a_position;
}
