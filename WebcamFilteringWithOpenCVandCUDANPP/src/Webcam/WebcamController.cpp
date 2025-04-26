#include "WebcamController.h"

#include <iostream>

#include <opencv4/opencv2/core/cuda.hpp>


WebcamController::WebcamController()
{
	initVideoCapture();
}

void WebcamController::startVideoCapture()
{
	videoCaptureThread = std::jthread(&WebcamController::startVideoCaptureThread, this);
}

bool WebcamController::getCameraFrame(cv::Mat& mat)
{
	if (currentCamFrame.empty() == false)
	{
		// filterMutexes.at(GPUMatTypes::CamFrame).lock();

		mat = currentCamFrame;

		// filterMutexes.at(GPUMatTypes::CamFrame).unlock();

		return true;
	}

	return false;
}

void WebcamController::setActiveFilter(bool activate, FilterType filterType)
{
	if (activate)
	{
		bool filterInserted = false;

		const auto& filteredFrame = filteredFrames.find(filterType);
		if (filteredFrame == filteredFrames.end())
		{
			filteredFrames[filterType] = std::make_shared<cv::Mat>(currentCamFrame.size(), currentCamFrame.type());
			filterInserted = true;
		}

		if (filterInserted)
		{
			if (filteredFrames.size() == 1)
			{
				// Set the initial gpu frames
				gpuMatFrames[GPUMatTypes::CamFrame] = std::make_shared<cv::cuda::GpuMat>();
				gpuMatFrames.at(GPUMatTypes::CamFrame)->create(currentCamFrame.size(), currentCamFrame.type());
			}

			switch (filterType)
			{
				case FilterType::Grayscale:
					gpuMatFrames[GPUMatTypes::GrayFrame] = std::make_shared<cv::cuda::GpuMat>();
					gpuMatFrames[GPUMatTypes::GrayFrameRGB] = std::make_shared<cv::cuda::GpuMat>();
					break;
				case FilterType::Sobel:
					gpuMatFrames[GPUMatTypes::SobelFrame] = std::make_shared<cv::cuda::GpuMat>();
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

bool WebcamController::getFilteredFrame(FilterType filterType, std::shared_ptr<cv::Mat>& mat)
{
	const auto& filteredFrame = filteredFrames.find(filterType);
	if (filteredFrame == filteredFrames.end())
	{
		mat = filteredFrame->second;
		return true;
	}

	return false;
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
	filterMutexes[GPUMatTypes::CamFrame];
	currentCamFrame = cv::Mat(cameraHeight, cameraWidth, CV_8UC3);


	filterMutexes.at(GPUMatTypes::CamFrame).lock();

	camCapture >> currentCamFrame;

	filterMutexes.at(GPUMatTypes::CamFrame).unlock();

	if (currentCamFrame.empty())
	{
		std::cout << "Error: Could not capture frame. \n";
		return;
	}
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
		filterMutexes.at(GPUMatTypes::CamFrame).lock();

		camCapture >> currentCamFrame;

		filterMutexes.at(GPUMatTypes::CamFrame).unlock();

		if (currentCamFrame.empty())
		{
			std::cout << "Error: Could not capture frame. \n";
			return;
		}

		for (const auto& filter : filteredFrames)
		{
			switch (filter.first)
			{
				case FilterType::Grayscale:
					generateGrayscaleRGBFrame();
					break;
				case FilterType::Sobel:
					generateSobelFilteredFrame();
					break;
				default:
					break;
			}
		}
	}
}

void WebcamController::generateGrayscaleRGBFrame()
{

}

void WebcamController::generateSobelFilteredFrame()
{
}
