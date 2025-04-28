#include "WebcamView.h"

#include <iostream>

#include <GL/gl3w.h>

#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>

#include "../Texture/ImageTexture.h"


WebcamView::WebcamView() :
	webcamController(WebcamController())
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

void WebcamView::showMainContents()
{
	ImGui::Begin("Main");

	ImGui::SliderFloat("gain", &gain, 0.0f, 2.0f, "%.3f");

	ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
				ImGui::GetIO().Framerate);

	addFiltersTable();

	ImGui::End();
}

void WebcamView::addFiltersTable()
{
	ImGui::BeginTable("Filters", 1, ImGuiTableFlags_BordersOuter);

	ImGui::TableSetupColumn("Filters");
	ImGui::TableHeadersRow();

	for (auto& filter : webcamController.activeFilters)
	{
		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);

		ImGui::PushID((int)filter.first);

		if (ImGui::Checkbox(webcamController.activeFiltersStrings.at(filter.first).c_str(), &filter.second))
		{
			webcamController.setActiveFilter(filter.first, filter.second);
		}

		ImGui::PopID();
	}

	ImGui::EndTable();
}

void WebcamView::show()
{
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplSDL2_NewFrame();

	ImGui::NewFrame();

	showMainContents();

	std::vector<std::shared_ptr<ImageTexture>> textures;

	for (const auto& filter : webcamController.activeFilters)
	{
		if (filter.second == false)// inactive
			continue;

		std::string window_name = webcamController.activeFiltersStrings.at(filter.first);

		const cv::Mat* frame = nullptr;

		webcamController.setFrameMutexLocked(filter.first, true);
		webcamController.getFilteredFrame(filter.first, frame);

		std::shared_ptr<ImageTexture>& texture = textures.emplace_back(std::make_shared<ImageTexture>());
		texture->setImage(frame);

		webcamController.setFrameMutexLocked(filter.first, false);

		ImGui::Begin(window_name.c_str());

		ImGui::Image((ImTextureID)(intptr_t)texture->getOpenglTexture(), texture->getSize());

		ImGui::End();
	}

	render();
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
	while (!handleEvent())
	{
		show();
	}

	exit();
}