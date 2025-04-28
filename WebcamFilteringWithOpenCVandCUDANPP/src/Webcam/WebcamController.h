#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <nppdefs.h>

#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/core/mat.hpp>
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

	void setActiveFilter(FilterType filterType, bool active);
	bool getFilteredFrame(FilterType filterType, const cv::Mat*& mat);

	void setFrameMutexLocked(FilterType filterType, bool lock);

	std::unordered_map<FilterType, bool> activeFilters;
	std::unordered_map<FilterType, std::string> activeFiltersStrings;

private:
	void initVideoCapture();
	void startVideoCaptureThread();


	void generateCameraFrame();
	void generateGrayscaleRGBFrame();
	void generateSobelFilteredFrame();

	cv::VideoCapture camCapture;
	cv::Mat currentCamFrame;

	bool videoCaptureCanBeStarted;
	std::jthread videoCaptureThread;


	std::unordered_map<FilterType, cv::Mat> filteredFrames;
	std::unordered_map<FilterType, std::mutex> filterMutexes;

	enum class GPUMatTypes
	{
		CamFrame,
		GrayFrame,
		GrayFrameRGB,
		SobelFrame,
		CurrentFiltersCombined
	};
	std::unordered_map<GPUMatTypes, cv::cuda::GpuMat> gpuMatFrames;

	cv::cuda::GpuMat gradXGpu;
	cv::cuda::GpuMat gradYGpu;
};

