#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

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
	bool getFilteredFrame(FilterType filterType, cv::Mat*& mat);

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
	std::jthread videoCaptureThread;
	cv::Mat currentCamFrame;

	std::unordered_map<FilterType, cv::Mat> filteredFrames;
	std::unordered_map<FilterType, std::mutex> filterMutexes;

	enum class GPUMatTypes
	{
		CamFrame,
		GrayFrame,
		GrayFrameRGB,
		SobelFrame
	};
	std::unordered_map<GPUMatTypes, cv::cuda::GpuMat> gpuMatFrames;
};

