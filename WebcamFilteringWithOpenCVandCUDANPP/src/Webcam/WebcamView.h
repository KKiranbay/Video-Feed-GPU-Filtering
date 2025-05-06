#pragma once

#include <string>
#include <vector>

#include <imgui.h>

#include <SDL2/SDL.h>

#include <opencv4/opencv2/core/mat.hpp>

#include "EventQueues/ViewEventQueue.h"
#include "WebcamController.h"


class WebcamView
{
	friend class WebcamController;

public:
	WebcamView();

	void startMainLoop();

protected:
	std::shared_ptr<ViewEvent> getEventFromQueue();

private:
	void init();
	void initContents();
	void render();
	void showMainContents();

	bool handleEvent();

	void show();
	void exit();
	float getGain();

	void addFiltersTable();
	void addFilterRow(FilterTypeEnum filterType);

	void addEventToQueue(std::shared_ptr<ViewEvent> viewEvent);

	// Event functions
	void onActivateCombinedFilterClicked();
	void onActiveFilterComboboxClicked(const FilterTypeEnum& filterType, const bool& isActive);
	void onActiveFilterOnCombinedFilterComboboxClicked(const FilterTypeEnum& filterType, const bool& isAdded);

	// Variables
	WebcamController webcamController;

	bool windowCreated;
	std::string m_windowName;

	SDL_Window* window;
	SDL_GLContext gl_context;
	ImGuiIO* io;

	ImVec4 clear_color;

	float gain;

	ViewEventQueue m_ViewEventQueue;

	bool m_combinedFiltersActive;

	std::unordered_map<FilterTypeEnum, bool> m_activeFiltersMap;
	std::unordered_map<FilterTypeEnum, std::string> m_activeFiltersStrings;

	std::unordered_map<FilterTypeEnum, bool> m_combinedFilters;
};

