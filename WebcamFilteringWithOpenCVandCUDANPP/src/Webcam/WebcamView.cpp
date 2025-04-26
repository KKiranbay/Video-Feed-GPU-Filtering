#include "WebcamView.h"

#include <iostream>

#include <GL/gl3w.h>

#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>

#include "../Texture/ImageTexture.h"


WebcamView::WebcamView()
{
	init();
	initContents();

	io = &ImGui::GetIO();
	(void)&io;

	// dynamic contents
	gain = 1.0f;

	webcamController.startVideoCapture();
}

float WebcamView::getGain()
{
	return gain;
}

void WebcamView::init()
{
	// Setup SDL
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) !=
		0)
	{
		std::cout << "Error: %s\n" << SDL_GetError() << "\n";
		exit();
		return;
	}

	const char* glsl_version = "#version 150";
	SDL_GL_SetAttribute(
		SDL_GL_CONTEXT_FLAGS,
		SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);  // Always required on Mac
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);

	// Create window with graphics context
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
	SDL_WindowFlags window_flags = (SDL_WindowFlags)(
		SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
	// SDL_Window*
	window = SDL_CreateWindow("OpenCV/ImGUI Viewer", SDL_WINDOWPOS_CENTERED,
							  SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
	// SDL_GLContext
	gl_context = SDL_GL_CreateContext(window);
	SDL_GL_SetSwapInterval(1);  // Enable vsync

	bool err = gl3wInit() != 0;
	if (err)
	{
		fprintf(stderr, "Failed to initialize OpenGL loader!\n");
		exit();
	}

	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();

	// Setup Platform/Renderer bindings
	ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
	ImGui_ImplOpenGL3_Init(glsl_version);
}

void WebcamView::initContents()
{
	clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
}

void WebcamView::exit()
{
	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplSDL2_Shutdown();
	ImGui::DestroyContext();

	SDL_GL_DeleteContext(gl_context);
	SDL_DestroyWindow(window);
	SDL_Quit();
}

void WebcamView::render()
{
	// Rendering
	ImGui::Render();
	SDL_GL_MakeCurrent(window, gl_context);
	glViewport(0, 0, (int)io->DisplaySize.x, (int)io->DisplaySize.y);
	glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
	glClear(GL_COLOR_BUFFER_BIT);
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	SDL_GL_SwapWindow(window);
}

void WebcamView::imshow(std::string frame_name, cv::Mat* frame)
{
	frame_names.push_back(frame_name);
	frames.push_back(frame);
}

void WebcamView::imshow(cv::Mat* frame)
{
	imshow("image:" + std::to_string(frames.size()), frame);
}

void WebcamView::showMainContents()
{
	ImGui::Begin("Main");

	ImGui::SliderFloat("gain", &gain, 0.0f, 2.0f, "%.3f");

	ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
				ImGui::GetIO().Framerate);
	ImGui::End();
}

void WebcamView::show()
{
	static cv::Mat camFrame;

	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplSDL2_NewFrame();

	ImGui::NewFrame();

	showMainContents();

	if (webcamController.getCameraFrame(camFrame))
	{
		imshow("Webcam", &camFrame);
	}

	// initialize textures
	std::vector<ImageTexture*> my_textures;
	for (int i = 0; i < frames.size(); i++)
	{
		my_textures.push_back(new ImageTexture());
	}

	// imshow windows
	for (int i = 0; i < frames.size(); i++)
	{
		cv::Mat* frame = frames[i];

		std::string window_name;
		if (frame_names.size() <= i)
		{
			window_name = "image:" + std::to_string(i);
		}
		else
		{
			window_name = frame_names[i];
		}

		ImGui::Begin(window_name.c_str());

		my_textures[i]->setImage(frame);
		ImGui::Image((ImTextureID)(intptr_t)my_textures[i]->getOpenglTexture(), my_textures[i]->getSize());

		ImGui::End();
	}

	render();

	// clear resources
	for (int i = 0; i < frames.size(); i++)
	{
		delete my_textures[i];
	}

	frame_names.clear();
	frames.clear();
	my_textures.clear();
}

bool WebcamView::handleEvent()
{
	SDL_Event event;
	bool done = false;
	while (SDL_PollEvent(&event))
	{
		ImGui_ImplSDL2_ProcessEvent(&event);
		if (event.type == SDL_QUIT) done = true;
		if (event.type == SDL_WINDOWEVENT &&
			event.window.event == SDL_WINDOWEVENT_CLOSE &&
			event.window.windowID == SDL_GetWindowID(window))
			done = true;
	}
	return done;
}

void WebcamView::startMainLoop()
{
	// Main loop
	while (!handleEvent())
	{
		show();
	}

	exit();
}