#version 330 core
precision mediump float;

in vec2 v_texCoord;
out vec4 outColor;

uniform sampler2D s_texture;

void main() {
	outColor = texture( s_texture, v_texCoord );
	//outColor = vec4(0.999f, 0.111f, 0.111f, 1.0f);
}