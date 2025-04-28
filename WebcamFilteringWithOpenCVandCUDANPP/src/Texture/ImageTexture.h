#pragma once

#include <gl/gl3w.h>
#include <imgui.h>

#include <opencv4/opencv2/core/mat.hpp>

class ImageTexture
{
public:
	~ImageTexture();

	void setImage(const cv::Mat* frame);
	void* getOpenglTexture();
	ImVec2 getSize();

private:
	int width, height;
	GLuint m_opengl_texture;
};

