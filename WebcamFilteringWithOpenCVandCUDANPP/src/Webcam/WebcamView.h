#pragma once

#include <SDL2/SDL.h>

#include "EventQueues/ViewEventQueue.h"
#include "Texture/ImageTexture.h"
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
	void showFilters();
	void clearTextures();

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

	// View Variables
	SDL_Window* window;
	int m_windowWidht;
	int m_windowHeight;
	float m_mainContentsWidth;

	SDL_GLContext gl_context;
	ImGuiIO* io;

	ImVec4 clear_color;
	float gain;

	// Controller Variables
	ViewEventQueue m_ViewEventQueue;

	WebcamController m_WebcamController;

	WebcamMats m_ViewsWebcamMats;

	bool m_View_CombinedFiltersActive;

	std::unordered_map<FilterTypeEnum, bool> m_View_ActiveFiltersMap;
	std::unordered_map<FilterTypeEnum, std::string> m_View_ActiveFiltersStrings;

	std::unordered_map<FilterTypeEnum, bool> m_View_CombinedFilters;

	std::vector<ImageTexture> m_FilteredTextures;
	ImageTexture m_CombinedTexture;
};

