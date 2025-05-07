#pragma once

#include <memory>
#include <mutex>
#include <thread>

#include <nppdefs.h>

#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/videoio.hpp>

#include "Filters/FilterTypes.h"
#include "WebcamMats.h"

class ViewEvent;
class WebcamView;


class WebcamController
{
public:
	WebcamController(WebcamView* parentView);

	void startVideoCapture();

	void getMats(WebcamMats& webcamMatsFromView);

	int activeFiltersCount;
	std::unordered_map<FilterTypeEnum, bool> activeFiltersMap;

	bool combinedFiltersActive;
	std::unordered_map<FilterTypeEnum, bool> combinedFilters;

private:
	void initVariables();
	void initGpuMatsAndMutexesMap();

	void initVideoCapture();
	void startVideoCaptureThread();

	void processEvents();

	void flipCameraFrame();
	void generateActiveFilters();

	void generateCameraFrame();
	void generateGrayscaleRGBFrame();
	void generateSobelFilteredFrame();
	void generateCombinedFilteredFrame();

	void combinedFrameInitOrDestroy();

	void processChangedActiveFilters(std::shared_ptr<ViewEvent> event);
	void processChangedCombinedFiltersActive(std::shared_ptr<ViewEvent> event);
	void processChangedActiveFiltersOnCombinedFilters(std::shared_ptr<ViewEvent> event);

	void changeActiveCombinedFilters(FilterTypeEnum filterType, bool isActive);

	enum class GPUMatTypesEnum
	{
		CamFrame,
		GrayFrame,
		GrayFrameRGB,
		SobelFrame,
		SobelGradXGpu,
		SobelGradYGpu,
		CurrentFiltersCombined
	};

	// Variables
	WebcamView* parentView;

	WebcamMats m_ControllersWebcamMats;
	std::mutex m_WebcamMatsMutex;

	cv::VideoCapture camCapture;
	cv::Mat currentCamFrame;

	bool videoCaptureCanBeStarted;
	std::jthread videoCaptureThread;

	int combinedFiltersCount;

	std::unordered_map<GPUMatTypesEnum, cv::cuda::GpuMat> gpuMatsMap;
};

