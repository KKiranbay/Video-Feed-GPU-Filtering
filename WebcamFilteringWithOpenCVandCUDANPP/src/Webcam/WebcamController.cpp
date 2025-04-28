#include "WebcamController.h"

#include <iostream>

#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_color_conversion.h>
#include <nppi_filtering_functions.h>

#include <opencv4/opencv2/core/cuda.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>


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

	videoCaptureCanBeStarted = false;

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

	videoCaptureCanBeStarted = true;
}

void WebcamController::startVideoCapture()
{
	if (videoCaptureCanBeStarted)
		videoCaptureThread = std::jthread(&WebcamController::startVideoCaptureThread, this);
}

void WebcamController::startVideoCaptureThread()
{
	while (true)
	{
		camCapture >> currentCamFrame;
		if (currentCamFrame.empty())
		{
			std::cout << "Error: Could not capture frame. \n";
			return;
		}

		if (filteredFrames.empty() == false)
		{
			gpuMatFrames.at(GPUMatTypes::CamFrame).upload(currentCamFrame);

			cv::cuda::GpuMat& camFrameGpuMat = gpuMatFrames.at(GPUMatTypes::CamFrame);

			cv::cuda::flip(camFrameGpuMat, camFrameGpuMat, 1);
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
	gpuMatFrames.at(GPUMatTypes::CamFrame).download(filteredFrames.at(FilterType::None));

	/*gpuMatFrames.at(GPUMatTypes::CamFrame).copyTo(gpuMatFrames.at(GPUMatTypes::CurrentFiltersCombined)
					 .rowRange(0, gpu_frame.size().height)
					 .colRange(0, gpu_frame.size().width));*/
}

void WebcamController::generateGrayscaleRGBFrame()
{
	cv::cuda::GpuMat& camFrameGpuMat = gpuMatFrames.at(GPUMatTypes::CamFrame);
	cv::cuda::GpuMat& grayFrameGpuMat = gpuMatFrames.at(GPUMatTypes::GrayFrame);

	NppStatus status = nppiRGBToGray_8u_C3C1R(camFrameGpuMat.ptr(), static_cast<int>(camFrameGpuMat.step),
											  grayFrameGpuMat.ptr(), static_cast<int>(grayFrameGpuMat.step),
											  { camFrameGpuMat.cols, camFrameGpuMat.rows });
	if (status != NPP_SUCCESS)
	{
		std::cerr << "Error computing grayscale gradient: " << status << std::endl;
		return;
	}

	cv::cuda::GpuMat& grayFrameRGBGpuMat = gpuMatFrames.at(GPUMatTypes::GrayFrameRGB);

	cv::cuda::cvtColor(grayFrameGpuMat, grayFrameRGBGpuMat, cv::COLOR_GRAY2RGB);

	grayFrameRGBGpuMat.download(filteredFrames.at(FilterType::Grayscale));

	/*grayFrameRGBGpuMat.copyTo(gpuMatFrames.at(GPUMatTypes::CurrentFiltersCombined)
							  .rowRange(0, grayFrameRGBGpuMat.size().height)
							  .colRange(grayFrameRGBGpuMat.size().width, grayFrameRGBGpuMat.size().width * 2));*/
}

void WebcamController::generateSobelFilteredFrame()
{
	cv::cuda::GpuMat& camFrameGpuMat = gpuMatFrames.at(GPUMatTypes::CamFrame);

	const Npp8u* camFrameGpuMatPtr = static_cast<const Npp8u*>(camFrameGpuMat.ptr());
	Npp8u* gradXGpuPtr = static_cast<Npp8u*>(gradXGpu.ptr());
	NppiSize roi = { camFrameGpuMat.cols, camFrameGpuMat.rows };

	NppStatus status = nppiFilterSobelHoriz_8u_C3R(camFrameGpuMatPtr, static_cast<Npp32s>(camFrameGpuMat.step),
												   gradXGpuPtr, static_cast<Npp32s>(gradXGpu.step),
												   roi);
	if (status != NPP_SUCCESS)
	{
		std::cerr << "Error computing horizontal gradient: " << status << std::endl;
		return;
	}

	Npp8u* gradYGpuPtr = static_cast<Npp8u*>(gradYGpu.ptr());

	status = nppiFilterSobelVert_8u_C3R(camFrameGpuMatPtr, static_cast<Npp32s>(camFrameGpuMat.step),
										gradYGpuPtr, static_cast<Npp32s>(gradYGpu.step),
										roi);
	if (status != NPP_SUCCESS)
	{
		std::cerr << "Error computing vertical gradient: " << status << std::endl;
		return;
	}

	cv::cuda::GpuMat& sobelFrameGpuMat = gpuMatFrames.at(GPUMatTypes::SobelFrame);

	status = nppiAdd_8u_C3RSfs(gradXGpuPtr, static_cast<int>(gradXGpu.step),
							   gradYGpuPtr, static_cast<int>(gradYGpu.step),
							   static_cast<Npp8u*>(sobelFrameGpuMat.ptr()), static_cast<int>(sobelFrameGpuMat.step),
							   roi, 0); // no scaling
	if (status != NPP_SUCCESS)
	{
		std::cerr << "Error computing magnitude: " << status << std::endl;
		return;
	}

	sobelFrameGpuMat.download(filteredFrames.at(FilterType::Sobel));
}

void WebcamController::setActiveFilter(FilterType filterType, bool active)
{
	std::lock_guard<std::mutex> lock(filterMutexes[filterType]);

	const auto& endItr = filteredFrames.end();

	if (active)
	{
		const auto& filteredFrame = filteredFrames.find(filterType);
		if (filteredFrame == endItr)
		{
			filteredFrames[filterType] = cv::Mat(currentCamFrame.size(), currentCamFrame.type());
		}
		else
			return;

		size_t filteredFramesSize = filteredFrames.size();
		if (filteredFramesSize == 1)
		{
			gpuMatFrames[GPUMatTypes::CamFrame] = cv::cuda::GpuMat();
			gpuMatFrames.at(GPUMatTypes::CamFrame).create(currentCamFrame.size(), currentCamFrame.type());
		}

		switch (filterType)
		{
			case FilterType::Grayscale:
			{
				gpuMatFrames[GPUMatTypes::GrayFrame] = cv::cuda::GpuMat();
				gpuMatFrames[GPUMatTypes::GrayFrameRGB] = cv::cuda::GpuMat();

				gpuMatFrames.at(GPUMatTypes::GrayFrame).create(currentCamFrame.size(), CV_8UC1);
				gpuMatFrames.at(GPUMatTypes::GrayFrameRGB).create(currentCamFrame.size(), currentCamFrame.type());

				break;
			}
			case FilterType::Sobel:
			{
				gpuMatFrames[GPUMatTypes::SobelFrame] = cv::cuda::GpuMat();

				gpuMatFrames.at(GPUMatTypes::SobelFrame).create(currentCamFrame.size(), currentCamFrame.type());

				gradXGpu.create(currentCamFrame.size(), currentCamFrame.type());
				gradYGpu.create(currentCamFrame.size(), currentCamFrame.type());

				break;
			}
			default:
				break;
		}
	}
	else
	{
		if (filteredFrames.erase(filterType))
		{
			if (filteredFrames.empty())
			{
				gpuMatFrames.erase(GPUMatTypes::CamFrame);
			}

			switch (filterType)
			{
				case FilterType::Grayscale:
				{
					gpuMatFrames.erase(GPUMatTypes::GrayFrame);
					gpuMatFrames.erase(GPUMatTypes::GrayFrameRGB);
					break;
				}
				case FilterType::Sobel:
				{
					gpuMatFrames.erase(GPUMatTypes::SobelFrame);

					gradXGpu.release();
					gradYGpu.release();

					break;
				}
				default:
					break;
			}
		}
	}
}

bool WebcamController::getFilteredFrame(FilterType filterType, const cv::Mat*& mat)
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
