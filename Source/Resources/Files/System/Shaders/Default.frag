#version 330 core
precision mediump float;

uniform sampler2D s_texture;

in vec2 v_texCoord;
out vec4 outColor;

void main() {
	outColor = texture2D(s_texture, v_texCoord);
}
