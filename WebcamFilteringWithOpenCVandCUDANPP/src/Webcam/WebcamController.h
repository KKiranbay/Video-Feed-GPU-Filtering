#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/videoio.hpp>


enum class FilterType
{
	Grayscale,
	Sobel
};

class WebcamController
{
public:
	WebcamController();

	void startVideoCapture();

	bool getCameraFrame(cv::Mat& mat);

	void setActiveFilter(bool activate, FilterType filterType);
	bool getFilteredFrame(FilterType filterType, std::shared_ptr<cv::Mat>& mat);

private:
	void initVideoCapture();
	void startVideoCaptureThread();

	void generateGrayscaleRGBFrame();
	void generateSobelFilteredFrame();

	cv::VideoCapture camCapture;
	std::jthread videoCaptureThread;

	cv::Mat currentCamFrame;

	std::unordered_map<FilterType, std::shared_ptr<cv::Mat>> filteredFrames;

	enum class GPUMatTypes
	{
		CamFrame,
		GrayFrame,
		GrayFrameRGB,
		SobelFrame
	};
	std::unordered_map<GPUMatTypes, std::shared_ptr<cv::cuda::GpuMat>> gpuMatFrames;

	std::unordered_map<GPUMatTypes, std::mutex> filterMutexes;
};

