#include "WebcamController.h"

#include <iostream>

#include <opencv4/opencv2/core/cuda.hpp>


WebcamController::WebcamController()
{
	activeFilters = {
		{ FilterType::None, false },
		{ FilterType::Grayscale, false },
		{ FilterType::Sobel, false }
	};

	activeFiltersStrings = {
		{ FilterType::None, "None" },
		{ FilterType::Grayscale, "Grayscale" },
		{ FilterType::Sobel, "Sobel" }
	};

	initVideoCapture();
}

void WebcamController::initVideoCapture()
{
	camCapture = cv::VideoCapture(0, cv::CAP_DSHOW);

	int cameraWidth = 1280;
	int cameraHeight = 720;
	int cameraFps = 60;

	camCapture.set(cv::CAP_PROP_FRAME_WIDTH, cameraWidth);
	camCapture.set(cv::CAP_PROP_FRAME_HEIGHT, cameraHeight);

	camCapture.set(cv::CAP_PROP_FPS, cameraFps);

	camCapture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')); // MJPG

	if (!camCapture.isOpened())
	{
		std::cout << "Error: Could not open camera. \n";
		return;
	}

	double actualWidth = camCapture.get(cv::CAP_PROP_FRAME_WIDTH);
	double actualHeight = camCapture.get(cv::CAP_PROP_FRAME_HEIGHT);
	double actualFps = camCapture.get(cv::CAP_PROP_FPS);

	std::cout
		<< "Camera capture initialized!"
		<< "-----------------------------------------\n"
		<< "Actual capture resolution:\n"
		<< actualWidth << "x" << actualHeight << " @ " << actualFps << " fps\n"
		<< "-----------------------------------------";

	// Set the initial camera frame
	camCapture >> currentCamFrame;

	if (currentCamFrame.empty())
	{
		std::cout << "Error: Could not capture frame. \n";
		return;
	}
}

void WebcamController::startVideoCapture()
{
	videoCaptureThread = std::jthread(&WebcamController::startVideoCaptureThread, this);
}

void WebcamController::startVideoCaptureThread()
{
	if (!camCapture.isOpened())
	{
		std::cout << "Error: Could not open camera. \n";
		return;
	}

	while (true)
	{
		camCapture >> currentCamFrame;
		if (currentCamFrame.empty())
		{
			std::cout << "Error: Could not capture frame. \n";
			return;
		}

		for (const auto& filter : filteredFrames)
		{
			std::lock_guard<std::mutex> lock(filterMutexes[filter.first]);

			switch (filter.first)
			{
			case FilterType::None:
			{
				generateCameraFrame();
				break;
			}
			case FilterType::Grayscale:
			{
				generateGrayscaleRGBFrame();
				break;
			}
			case FilterType::Sobel:
			{
				generateSobelFilteredFrame();
				break;
			}
			default:
				break;
			}
		}
	}
}

void WebcamController::generateCameraFrame()
{
	filteredFrames.at(FilterType::None) = currentCamFrame;
}

void WebcamController::generateGrayscaleRGBFrame()
{
	filteredFrames.at(FilterType::Grayscale) = currentCamFrame;
}

void WebcamController::generateSobelFilteredFrame()
{
	filteredFrames.at(FilterType::Sobel) = currentCamFrame;
}

void WebcamController::setActiveFilter(FilterType filterType, bool active)
{
	std::lock_guard<std::mutex> lock(filterMutexes[filterType]);

	if (active)
	{
		bool filterInserted = false;

		const auto& filteredFrame = filteredFrames.find(filterType);
		if (filteredFrame == filteredFrames.end())
		{
			filteredFrames[filterType] = cv::Mat(currentCamFrame.size(), currentCamFrame.type());
			filterInserted = true;
		}

		if (filterInserted)
		{
			if (filteredFrames.size() == 1)
			{
				// Set the initial gpu frames
				gpuMatFrames[GPUMatTypes::CamFrame] = cv::cuda::GpuMat();
				gpuMatFrames.at(GPUMatTypes::CamFrame).create(currentCamFrame.size(), currentCamFrame.type());
			}

			switch (filterType)
			{
			case FilterType::Grayscale:
				gpuMatFrames[GPUMatTypes::GrayFrame] = cv::cuda::GpuMat();
				gpuMatFrames[GPUMatTypes::GrayFrameRGB] = cv::cuda::GpuMat();
				break;
			case FilterType::Sobel:
				gpuMatFrames[GPUMatTypes::SobelFrame] = cv::cuda::GpuMat();
				break;
			default:
				break;
			}
		}
	}
	else
	{
		if (filteredFrames.erase(filterType) > 0)
		{
			switch (filterType)
			{
			case FilterType::Grayscale:
				gpuMatFrames.erase(GPUMatTypes::GrayFrame);
				gpuMatFrames.erase(GPUMatTypes::GrayFrameRGB);
				break;
			case FilterType::Sobel:
				gpuMatFrames.erase(GPUMatTypes::SobelFrame);
				break;
			default:
				break;
			}

			if (filteredFrames.empty())
			{
				gpuMatFrames.erase(GPUMatTypes::CamFrame);
			}
		}
	}
}

bool WebcamController::getFilteredFrame(FilterType filterType, cv::Mat*& mat)
{
	const auto& filteredFrame = filteredFrames.find(filterType);
	if (filteredFrame != filteredFrames.end())
	{
		mat = &filteredFrame->second;
		return true;
	}

	return false;
}

void WebcamController::setFrameMutexLocked(FilterType filterType, bool lock)
{
	if (lock)
	{
		filterMutexes[filterType].lock();
	}
	else
	{
		filterMutexes[filterType].unlock();
	}
}
