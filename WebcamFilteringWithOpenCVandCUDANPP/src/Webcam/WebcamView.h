#pragma once

#include <string>
#include <vector>

#include <imgui.h>

#include <SDL2/SDL.h>

#include <opencv4/opencv2/core/mat.hpp>

#include "WebcamController.h"


class WebcamView
{
public:
	WebcamView();

	void startMainLoop();

	bool handleEvent();
	void imshow(std::string, cv::Mat*);
	void imshow(cv::Mat*);
	void show();
	void exit();
	float getGain();

private:
	void init();
	void initContents();
	void render();
	void showMainContents();

	WebcamController webcamController;

	bool windowCreated;
	std::string m_windowName;

	SDL_Window* window;
	SDL_GLContext gl_context;
	ImGuiIO* io;

	// static contents
	ImVec4 clear_color;

	std::vector<std::string> frame_names;
	std::vector<cv::Mat*> frames;
	float gain;
};

