#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

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

	const cv::Mat* getFilteredMat(FilterTypeEnum filterType);
	const cv::Mat* getCurrentFiltersCombinedFrame();
	WebcamMats getMats();

	int activeFiltersCount;
	std::unordered_map<FilterTypeEnum, bool> activeFiltersMap;

	bool combinedFiltersActive;
	std::unordered_map<FilterTypeEnum, bool> combinedFilters;

private:
	void initVariables();
	void initFilteredMatsAndMutexesMap();
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

	void changedActiveCombinedFilters(FilterTypeEnum filterType, bool isActive);

	WebcamMats webcamMats;
	std::mutex m_WebcamMatsMutex;

	struct MatAndMutex
	{
		cv::Mat mat;
		std::mutex matMutex;
	};

	struct GpuMatAndMutex
	{
		cv::cuda::GpuMat gpuMat;
		std::mutex gpuMutex;
	};

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

	cv::VideoCapture camCapture;
	cv::Mat currentCamFrame;

	bool videoCaptureCanBeStarted;
	std::jthread videoCaptureThread;

	std::unordered_map<FilterTypeEnum, MatAndMutex> filteredMatsAndMutexesMap;

	MatAndMutex currentFiltersCombinedMatAndMutex;

	std::mutex combinedFiltersCountMutex;
	int combinedFiltersCount;

	std::unordered_map<GPUMatTypesEnum, GpuMatAndMutex> gpuMatsAndMutexesMap;
};

