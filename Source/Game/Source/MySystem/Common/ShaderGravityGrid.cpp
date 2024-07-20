// ◦ Xyz ◦

#include "../Common/ShaderGravityGrid.h"
#include <glad/gl.h>
#include <FileManager.h>
#include <Draw/Camera/Camera.h>

unsigned int ShaderGravityGrid::u_matProjectionView = 0;
unsigned int ShaderGravityGrid::u_matViewModel = 0;
unsigned int ShaderGravityGrid::u_color = 0;
unsigned int ShaderGravityGrid::u_factor = 0;
unsigned int ShaderGravityGrid::u_mass_factor = 0;
unsigned int ShaderGravityGrid::u_range = 0;
unsigned int ShaderGravityGrid::u_rangeZ = 0;

unsigned int ShaderGravityGrid::u_splashPosition;
unsigned int ShaderGravityGrid::u_distances;
unsigned int ShaderGravityGrid::u_splashCount;

unsigned int ShaderGravityGrid::u_body_count = 0;
unsigned int ShaderGravityGrid::u_body_positions = 0;
unsigned int ShaderGravityGrid::u_body_massess = 0;
unsigned int ShaderGravityGrid::u_body_colors = 0;

// ShaderSpatialGrid
void ShaderGravityGrid::Use()
{
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

void ShaderGravityGrid::GetLocation()
{
	if (_program == 0) {
		return;
	}

	u_matProjectionView = glGetUniformLocation(_program, "u_matProjectionView");
	u_matViewModel = glGetUniformLocation(_program, "u_matViewModel");

	u_color = glGetUniformLocation(_program, "u_color");
	u_factor = glGetUniformLocation(_program, "u_factor");
	u_mass_factor = glGetUniformLocation(_program, "u_mass_factor");
	u_range = glGetUniformLocation(_program, "u_range");
	u_rangeZ = glGetUniformLocation(_program, "u_rangeZ");

	u_splashPosition = glGetUniformLocation(_program, "u_splashPosition");
	u_distances = glGetUniformLocation(_program, "u_distances");
	u_splashCount = glGetUniformLocation(_program, "u_splashCount");

	u_body_count = glGetUniformLocation(_program, "u_body_count");
	u_body_positions = glGetUniformLocation(_program, "u_body_positions");
	u_body_massess = glGetUniformLocation(_program, "u_body_massess");
	u_body_colors = glGetUniformLocation(_program, "u_body_colors");
}

void ShaderGravityGrid::SetPosition(const float* pos)
{
	static glm::mat4x4 mat(1.f);
	mat[3][0] = pos[0];
	mat[3][1] = pos[1];
	mat[3][2] = pos[2];
	glUniformMatrix4fv(u_matViewModel, 1, GL_FALSE, value_ptr(mat));
}
