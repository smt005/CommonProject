#version 330 core

layout (location = 0) in vec3 a_position;

uniform mat4 u_matProjectionView;
uniform mat4 u_matViewModel;

void main() {
	//gl_Position = vec4(a_position, 1.0);
	//gl_Position = u_matProjectionView * vec4(a_position, 1.0);
	gl_Position = u_matProjectionView * u_matViewModel * vec4(a_position, 1.0);
}
