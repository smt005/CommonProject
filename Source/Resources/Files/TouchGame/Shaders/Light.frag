//precision mediump float;

varying vec2 v_texCoord;
varying float v_dot;

uniform sampler2D s_baseMap;
uniform vec4 u_color;

void main(void) {
	vec2 texCoord = v_texCoord;	
	vec4 color = texture2D(s_baseMap, v_texCoord).bgra * u_color;	
	float ambient = 0.5;
	
	vec4 res_color = color * v_dot;
	//vec4 res_color = color * v_dot;// + color * ambient;
	//vec4 res_color = color * ambient;
	
	res_color.a = 1.0;
	
	gl_FragColor = res_color;
}
