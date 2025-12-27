#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>

using namespace std;

class Shader {
public:
	unsigned int ID;
	Shader(const char* vertexSource, const char* fragmentSource);
	void use();
	void setInt(string name, int value);
	void setMat4(string name, glm::mat4 mat);
	void setVec3(string name, glm::vec3 value);
};
