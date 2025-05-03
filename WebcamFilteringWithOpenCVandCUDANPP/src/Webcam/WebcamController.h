#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <nppdefs.h>

#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/videoio.hpp>

#define NOMINMAX
#include "../Texture/ImageTexture.h"


enum class FilterTypeEnum
{
	None,
	Grayscale,
	Sobel
};

class WebcamController
{
public:
	WebcamController();

	void startVideoCapture();

	void changedActiveFilter(FilterTypeEnum filterType);
	void changedCombinedFiltersActive();
	void changedActiveCombinedFilters(FilterTypeEnum filterType);

	const cv::Mat* getFilteredMat(FilterTypeEnum filterType);

	const cv::Mat* getCurrentFiltersCombinedFrame();

	int activeFiltersCount;
	std::unordered_map<FilterTypeEnum, bool> activeFiltersMap;
	std::unordered_map<FilterTypeEnum, std::string> activeFiltersStrings;

	bool combinedFiltersActive;
	std::unordered_map<FilterTypeEnum, bool> combinedFilters;

private:
	void initVariables();
	void initFilteredMatsAndMutexesMap();
	void initGpuMatsAndMutexesMap();

	void initVideoCapture();
	void startVideoCaptureThread();

	void generateCameraFrame();
	void generateGrayscaleRGBFrame();
	void generateSobelFilteredFrame();
	void generateCombinedFilteredFrame();

	void combinedFrameInitOrDestroy();

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

