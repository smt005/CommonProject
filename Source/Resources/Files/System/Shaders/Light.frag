//precision mediump float;

varying vec2 v_texCoord;
varying float v_dot;

uniform sampler2D s_baseMap;
uniform vec4 u_color;

void main(void) {
	vec2 texCoord = v_texCoord;	
	vec4 color = texture2D(s_baseMap, v_texCoord) * u_color;	
	vec4 res_color = color * v_dot;
	res_color.a = color.a;
	
	gl_FragColor = res_color;
}