#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <nppdefs.h>

#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/videoio.hpp>


enum class FilterType
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

	void changedActiveFilter(FilterType filterType);
	void changedCombinedFiltersActive();
	void changedActiveCombinedFilters(FilterType filterType);

	bool getFilteredMat(FilterType filterType, cv::Mat*& imageMat);

	const cv::Mat* getCurrentFiltersCombinedFrame();

	int activeFiltersCount;
	std::unordered_map<FilterType, bool> activeFiltersMap;
	std::unordered_map<FilterType, std::string> activeFiltersStrings;

	bool combinedFiltersActive;
	std::unordered_map<FilterType, bool> combinedFilters;

private:
	void initVideoCapture();
	void startVideoCaptureThread();

	void generateCameraFrame();
	void generateGrayscaleRGBFrame();
	void generateSobelFilteredFrame();
	void generateCombinedFilteredFrame();

	void combinedFrameInitOrDestroy();

	cv::VideoCapture camCapture;
	cv::Mat currentCamFrame;

	bool videoCaptureCanBeStarted;
	std::jthread videoCaptureThread;

	std::unordered_map<FilterType, std::mutex> filteredFramesMutexes;
	std::unordered_map<FilterType, cv::Mat> filteredFrames;

	std::mutex currentFiltersCombinedFrameMutex;
	cv::Mat currentFiltersCombinedFrame;

	enum class GPUMatTypes
	{
		CamFrame,
		GrayFrame,
		GrayFrameRGB,
		SobelFrame,
		SobelGradXGpu,
		SobelGradYGpu,
		CurrentFiltersCombined
	};
	std::unordered_map<GPUMatTypes, std::mutex> gpuMatsMutexes;
	std::unordered_map<GPUMatTypes, cv::cuda::GpuMat> gpuMats;

	int combinedFiltersCount;

};

