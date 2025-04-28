#include "ImageTexture.h"

ImageTexture::~ImageTexture()
{
	glBindTexture(GL_TEXTURE_2D, 0);
	glDeleteTextures(1, &m_opengl_texture);
}

void ImageTexture::setImage(const cv::Mat* frame)
{
	width = frame->cols;
	height = frame->rows;

	glGenTextures(1, &m_opengl_texture);
	glBindTexture(GL_TEXTURE_2D, m_opengl_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

	// Some environments do not support GP_BGR
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
				 GL_BGR, GL_UNSIGNED_BYTE, frame->data);
}

void* ImageTexture::getOpenglTexture()
{
	return (void*)(intptr_t)m_opengl_texture;
}

ImVec2 ImageTexture::getSize()
{
	return ImVec2(static_cast<float>(width), static_cast<float>(height));
}
