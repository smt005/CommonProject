#include "../Common/ShaderGravityGrid.h"
#include <glad/gl.h>
#include <FileManager.h>
#include <Draw/Camera/Camera.h>

unsigned int ShaderGravityGrid::u_matProjectionView = 0;
unsigned int ShaderGravityGrid::u_color = 0;
unsigned int ShaderGravityGrid::u_factor = 0;
unsigned int ShaderGravityGrid::u_range = 0;
unsigned int ShaderGravityGrid::u_rangeZ = 0;
unsigned int ShaderGravityGrid::u_body_position = 0;
unsigned int ShaderGravityGrid::u_body_color = 0;
unsigned int ShaderGravityGrid::u_body_count = 0;
unsigned int ShaderGravityGrid::u_body_positions = 0;
unsigned int ShaderGravityGrid::u_body_massess = 0;
unsigned int ShaderGravityGrid::u_body_colors = 0;

// ShaderSpatialGrid
void ShaderGravityGrid::Use() {
	glUseProgram(_program);
	glUniformMatrix4fv(u_matProjectionView, 1, GL_FALSE, Camera::GetLink().ProjectViewFloat());

	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glDisable(GL_TEXTURE_2D);
	glEnable(GL_COLOR);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glEnableVertexAttribArray(0);
}

void ShaderGravityGrid::GetLocation() {
	if (_program == 0) {
		return;
	}

	u_matProjectionView = glGetUniformLocation(_program, "u_matProjectionView");
	u_color = glGetUniformLocation(_program, "u_color");
	u_factor = glGetUniformLocation(_program, "u_factor");
	u_range = glGetUniformLocation(_program, "u_range");
	u_rangeZ = glGetUniformLocation(_program, "u_rangeZ");

	u_body_position = glGetUniformLocation(_program, "u_body_position");
	u_body_color = glGetUniformLocation(_program, "u_body_color");

	u_body_count = glGetUniformLocation(_program, "u_body_count");
	u_body_positions = glGetUniformLocation(_program, "u_body_positions");
	u_body_massess = glGetUniformLocation(_program, "u_body_massess");
	u_body_colors = glGetUniformLocation(_program, "u_body_colors");
}
