uniform mat4 u_matProjectionView;
uniform mat4 u_matViewModel;
uniform vec4 u_lightPos;

attribute vec4 a_position;
attribute vec2 a_texCoord;
attribute vec4 a_normal;

varying vec2 v_texCoord;
varying float v_dot;

void main(void) {
	v_texCoord = a_texCoord;

	vec4 pos = u_matViewModel * a_position;

	vec3 lightDirection = u_lightPos.xyz - pos.xyz;
	lightDirection = normalize(lightDirection);
	
	vec3 normal = normalize(a_normal.xyz);
	
	v_dot = dot(normal, lightDirection);
	v_dot = max(0.0, v_dot);
	v_dot = min(0.25, v_dot);
	
	//v_dot = v_dot * 0.5;
	v_dot = v_dot + 0.75;
	
	gl_Position = u_matProjectionView * u_matViewModel * a_position;
}
