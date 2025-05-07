#include "WebcamView.h"

#include <iostream>

#include <GL/gl3w.h>

#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>

#include "Events/ViewEvents/ActivateCombinedFilter.h"
#include "Events/ViewEvents/ViewChangeFilterEvents/ChangeActiveFilters.h"
#include "Events/ViewEvents/ViewChangeFilterEvents/ChangeActiveFiltersOnCombinedFilter.h"
#include "Texture/ImageTexture.h"


WebcamView::WebcamView() :
	webcamController(WebcamController(this))
{
	init();
	initContents();

	io = &ImGui::GetIO();
	(void)&io;

	// dynamic contents
	gain = 1.0f;

	webcamController.startVideoCapture();
	m_combinedFiltersActive = webcamController.combinedFiltersActive;
	m_activeFiltersMap = webcamController.activeFiltersMap;
	m_activeFiltersStrings = {
		{ FilterTypeEnum::None, "None" },
		{ FilterTypeEnum::Grayscale, "Grayscale" },
		{ FilterTypeEnum::Sobel, "Sobel" }
	};
	m_combinedFilters = webcamController.combinedFilters;
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

	if (ImGui::Checkbox("Combine Filters", &m_combinedFiltersActive))
	{
		onActivateCombinedFilterClicked();
	}

	addFiltersTable();

	ImGui::End();
}

void WebcamView::addFiltersTable()
{
	ImGui::BeginTable("Filters", 2, ImGuiTableFlags_BordersOuter);

	ImGui::TableSetupColumn("Filters");
	ImGui::TableSetupColumn("Add to Combined");

	ImGui::TableHeadersRow();

	addFilterRow(FilterTypeEnum::None);
	addFilterRow(FilterTypeEnum::Grayscale);
	addFilterRow(FilterTypeEnum::Sobel);

	ImGui::EndTable();
}

void WebcamView::addFilterRow(FilterTypeEnum filterType)
{
	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);

	ImGui::PushID((int)filterType);

	bool& activeFilter = m_activeFiltersMap.at(filterType);
	if (ImGui::Checkbox(m_activeFiltersStrings.at(filterType).c_str(), &activeFilter))
	{
		onActiveFilterComboboxClicked(filterType, activeFilter);
	}

	if (activeFilter)
	{
		ImGui::TableSetColumnIndex(1);

		bool& activeFilterOnCombined = m_combinedFilters.at(filterType);
		if (ImGui::Checkbox("Add", &activeFilterOnCombined))
		{
			onActiveFilterOnCombinedFilterComboboxClicked(filterType, activeFilterOnCombined);
		}
	}

	ImGui::PopID();
}

void WebcamView::show()
{
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplSDL2_NewFrame();

	ImGui::NewFrame();

	showMainContents();

	// TODO change it so that you get all the textures at once to not cause race condition
	WebcamMats webcamMats = webcamController.getMats();
	std::vector<ImageTexture> textures(webcamMats.activeMatsCount);
	auto textureItr = textures.begin();

	for (const auto& mat : webcamMats.filteredMatsMap)
	{
		if (mat.second.empty()) // inactive
			continue;

		const cv::Mat* imageMat = &mat.second;

		textureItr->setImage(imageMat);

		std::string& window_name = m_activeFiltersStrings.at(mat.first);

		ImGui::Begin(window_name.c_str());

		ImGui::Image((ImTextureID)(intptr_t)textureItr->getOpenglTexture(), textureItr->getSize());

		ImGui::End();

		textureItr++;
	}

	ImageTexture combinedTexture;

	if (webcamMats.currentFiltersCombinedMat.empty() == false)
	{
		const cv::Mat* imageMat = &webcamMats.currentFiltersCombinedMat;
		combinedTexture.setImage(imageMat);

		ImGui::Begin("Filters Combined");

		ImGui::Image((ImTextureID)(intptr_t)combinedTexture.getOpenglTexture(), combinedTexture.getSize());

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

void WebcamView::addEventToQueue(std::shared_ptr<ViewEvent> viewEvent)
{
	m_ViewEventQueue.pushViewEvent(viewEvent);
}

std::shared_ptr<ViewEvent> WebcamView::getEventFromQueue()
{
	return m_ViewEventQueue.popViewEvent();
}

void WebcamView::onActivateCombinedFilterClicked()
{
	std::shared_ptr<ActivateCombinedFilter> activateCombinedFilter = std::make_shared<ActivateCombinedFilter>();
	activateCombinedFilter->setActivateCombinedFilter(m_combinedFiltersActive);

	addEventToQueue(activateCombinedFilter);
}

void WebcamView::onActiveFilterComboboxClicked(const FilterTypeEnum& filterType, const bool& isActive)
{
	std::shared_ptr<ChangeActiveFilters> changeActiveFilters = std::make_shared<ChangeActiveFilters>();
	changeActiveFilters->setActiveFilterType(filterType, isActive);

	addEventToQueue(changeActiveFilters);
}

void WebcamView::onActiveFilterOnCombinedFilterComboboxClicked(const FilterTypeEnum& filterType, const bool& isAdded)
{
	std::shared_ptr<ChangeActiveFiltersOnCombinedFilter> changeActiveFiltersOnCombinedFilter = std::make_shared<ChangeActiveFiltersOnCombinedFilter>();
	changeActiveFiltersOnCombinedFilter->setActiveFilterTypeOnCombined(filterType, isAdded);

	addEventToQueue(changeActiveFiltersOnCombinedFilter);
}
