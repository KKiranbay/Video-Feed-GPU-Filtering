#include "WebcamController.h"

#include <iostream>

#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_color_conversion.h>
#include <nppi_filtering_functions.h>

#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>


WebcamController::WebcamController()
{
	activeFiltersCount = 0;
	activeFiltersMap = {
		{ FilterType::None, false },
		{ FilterType::Grayscale, false },
		{ FilterType::Sobel, false }
	};

	activeFiltersStrings = {
		{ FilterType::None, "None" },
		{ FilterType::Grayscale, "Grayscale" },
		{ FilterType::Sobel, "Sobel" }
	};

	combinedFiltersActive = false;
	combinedFiltersCount = 0;
	combinedFilters = {
		{ FilterType::None, false },
		{ FilterType::Grayscale, false },
		{ FilterType::Sobel, false }
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

		if (filteredFrames.empty())
		{
			continue;
		}

		cv::cuda::GpuMat& camFrameGpuMat = gpuMats.at(GPUMatTypes::CamFrame);

		{
			std::lock_guard<std::mutex> lockCamFrameGpuMat(gpuMatsMutexes[GPUMatTypes::CamFrame]);

			camFrameGpuMat.upload(currentCamFrame);
			cv::cuda::flip(camFrameGpuMat, camFrameGpuMat, 1);
		}

		for (const auto& filter : filteredFrames)
		{
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

		if (combinedFiltersActive && combinedFiltersCount != 0 && gpuMats.find(GPUMatTypes::CurrentFiltersCombined) != gpuMats.end())
		{
			generateCombinedFilteredFrame();
		}
	}
}

void WebcamController::generateCameraFrame()
{
	std::lock_guard<std::mutex> lockFilterMutexes(filteredFramesMutexes[FilterType::None]);

	std::lock_guard<std::mutex> lockCamFrameGpuMat(gpuMatsMutexes[GPUMatTypes::CamFrame]);
	gpuMats.at(GPUMatTypes::CamFrame).download(filteredFrames.at(FilterType::None));
}

void WebcamController::generateGrayscaleRGBFrame()
{
	cv::cuda::GpuMat& camFrameGpuMat = gpuMats.at(GPUMatTypes::CamFrame);
	cv::cuda::GpuMat& grayFrameGpuMat = gpuMats.at(GPUMatTypes::GrayFrame);

	gpuMatsMutexes[GPUMatTypes::GrayFrame].lock();

	{
		std::lock_guard<std::mutex> lockCamFrameGpuMat(gpuMatsMutexes[GPUMatTypes::CamFrame]);
		NppStatus status = nppiRGBToGray_8u_C3C1R(camFrameGpuMat.ptr(), static_cast<int>(camFrameGpuMat.step),
												  grayFrameGpuMat.ptr(), static_cast<int>(grayFrameGpuMat.step),
												  { camFrameGpuMat.cols, camFrameGpuMat.rows });

		if (status != NPP_SUCCESS)
		{
			gpuMatsMutexes[GPUMatTypes::GrayFrame].unlock();

			std::cerr << "Error computing grayscale gradient: " << status << std::endl;
			return;
		}
	}

	cv::cuda::GpuMat& grayFrameRGBGpuMat = gpuMats.at(GPUMatTypes::GrayFrameRGB);

	gpuMatsMutexes[GPUMatTypes::GrayFrameRGB].lock();

	cv::cuda::cvtColor(grayFrameGpuMat, grayFrameRGBGpuMat, cv::COLOR_GRAY2RGB);

	gpuMatsMutexes[GPUMatTypes::GrayFrame].unlock();

	std::lock_guard<std::mutex> lockFilterMutexes(filteredFramesMutexes[FilterType::Grayscale]);

	grayFrameRGBGpuMat.download(filteredFrames.at(FilterType::Grayscale));

	gpuMatsMutexes[GPUMatTypes::GrayFrameRGB].unlock();
}

void WebcamController::generateSobelFilteredFrame()
{
	cv::cuda::GpuMat& camFrameGpuMat = gpuMats.at(GPUMatTypes::CamFrame);

	const Npp8u* camFrameGpuMatPtr = static_cast<const Npp8u*>(camFrameGpuMat.ptr());

	cv::cuda::GpuMat& gradXGpuMat = gpuMats.at(GPUMatTypes::SobelGradXGpu);
	Npp8u* gradXGpuPtr = static_cast<Npp8u*>(gradXGpuMat.ptr());
	NppiSize roi = { camFrameGpuMat.cols, camFrameGpuMat.rows };

	gpuMatsMutexes[GPUMatTypes::CamFrame].lock();
	gpuMatsMutexes[GPUMatTypes::SobelGradXGpu].lock();

	NppStatus status = nppiFilterSobelHoriz_8u_C3R(camFrameGpuMatPtr, static_cast<Npp32s>(camFrameGpuMat.step),
												   gradXGpuPtr, static_cast<Npp32s>(gradXGpuMat.step),
												   roi);
	if (status != NPP_SUCCESS)
	{
		gpuMatsMutexes[GPUMatTypes::CamFrame].unlock();
		gpuMatsMutexes[GPUMatTypes::SobelGradXGpu].unlock();

		std::cerr << "Error computing horizontal gradient: " << status << std::endl;
		return;
	}

	cv::cuda::GpuMat& gradYGpuMat = gpuMats.at(GPUMatTypes::SobelGradYGpu);
	Npp8u* gradYGpuPtr = static_cast<Npp8u*>(gradYGpuMat.ptr());

	gpuMatsMutexes[GPUMatTypes::SobelGradYGpu].lock();

	status = nppiFilterSobelVert_8u_C3R(camFrameGpuMatPtr, static_cast<Npp32s>(camFrameGpuMat.step),
										gradYGpuPtr, static_cast<Npp32s>(gradYGpuMat.step),
										roi);

	gpuMatsMutexes[GPUMatTypes::CamFrame].unlock();

	if (status != NPP_SUCCESS)
	{
		gpuMatsMutexes[GPUMatTypes::SobelGradYGpu].unlock();

		std::cerr << "Error computing vertical gradient: " << status << std::endl;
		return;
	}

	cv::cuda::GpuMat& sobelFrameGpuMat = gpuMats.at(GPUMatTypes::SobelFrame);

	gpuMatsMutexes[GPUMatTypes::SobelFrame].lock();

	status = nppiAdd_8u_C3RSfs(gradXGpuPtr, static_cast<int>(gradXGpuMat.step),
							   gradYGpuPtr, static_cast<int>(gradYGpuMat.step),
							   static_cast<Npp8u*>(sobelFrameGpuMat.ptr()), static_cast<int>(sobelFrameGpuMat.step),
							   roi, 0); // no scaling

	gpuMatsMutexes[GPUMatTypes::SobelGradXGpu].unlock();
	gpuMatsMutexes[GPUMatTypes::SobelGradYGpu].unlock();

	if (status != NPP_SUCCESS)
	{
		gpuMatsMutexes[GPUMatTypes::SobelFrame].unlock();

		std::cerr << "Error computing magnitude: " << status << std::endl;
		return;
	}

	std::lock_guard<std::mutex> lockSobelFrameMutex(filteredFramesMutexes[FilterType::Sobel]);
	sobelFrameGpuMat.download(filteredFrames.at(FilterType::Sobel));

	gpuMatsMutexes[GPUMatTypes::SobelFrame].unlock();
}

void WebcamController::generateCombinedFilteredFrame()
{
	std::lock_guard<std::mutex> lockGpuMatMutex(gpuMatsMutexes[GPUMatTypes::CurrentFiltersCombined]);

	cv::cuda::GpuMat& currentFiltersCombinedFrameGpuMat = gpuMats.at(GPUMatTypes::CurrentFiltersCombined);
	int combinedFiltersPlace = 0;

	cv::cuda::GpuMat* gpuMat = nullptr;

	for (const auto& filter : combinedFilters)
	{
		if (filter.second == false)
			continue;

		std::mutex* gpuMatMutex = nullptr;

		switch (filter.first)
		{
			case FilterType::None:
			{
				gpuMat = &gpuMats.at(GPUMatTypes::CamFrame);
				gpuMatMutex = &gpuMatsMutexes[GPUMatTypes::CamFrame];
				break;
			}
			case FilterType::Grayscale:
			{
				gpuMat = &gpuMats.at(GPUMatTypes::GrayFrameRGB);
				gpuMatMutex = &gpuMatsMutexes[GPUMatTypes::GrayFrameRGB];
				break;
			}
			case FilterType::Sobel:
			{
				gpuMat = &gpuMats.at(GPUMatTypes::SobelFrame);
				gpuMatMutex = &gpuMatsMutexes[GPUMatTypes::SobelFrame];
				break;
			}
			default:
				break;
		}

		if (gpuMat == nullptr || gpuMatMutex == nullptr)
			continue;

		int gpuMatHeight = gpuMat->size().height;
		int gpuMatWidth = gpuMat->size().width;

		std::lock_guard<std::mutex> lockGpuMatMutex(*gpuMatMutex);

		gpuMat->copyTo(currentFiltersCombinedFrameGpuMat
					   .rowRange(0, gpuMatHeight)
					   .colRange(gpuMatWidth * combinedFiltersPlace, gpuMatWidth * (combinedFiltersPlace + 1)));

		combinedFiltersPlace++;
	}

	std::lock_guard<std::mutex> lockCurrentFiltersCombinedFrameMutex(currentFiltersCombinedFrameMutex);
	currentFiltersCombinedFrameGpuMat.download(currentFiltersCombinedFrame);
}

void WebcamController::changedActiveFilter(FilterType filterType)
{
	std::lock_guard<std::mutex> lockFilteredFramesMutex(filteredFramesMutexes[filterType]);

	const auto& endItr = filteredFrames.end();
	bool active = activeFiltersMap.find(filterType)->second;

	if (active)
	{
		const auto& filteredFrame = filteredFrames.find(filterType);
		if (filteredFrame == endItr)
		{
			activeFiltersCount++;
			filteredFrames[filterType] = cv::Mat(currentCamFrame.size(), currentCamFrame.type());
		}
		else
			return;

		size_t filteredFramesSize = filteredFrames.size();
		if (filteredFramesSize == 1)
		{
			std::lock_guard<std::mutex> lockCamFrameGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::CamFrame]);

			gpuMats[GPUMatTypes::CamFrame];
			gpuMats.at(GPUMatTypes::CamFrame).create(currentCamFrame.size(), currentCamFrame.type());
		}

		switch (filterType)
		{
			case FilterType::Grayscale:
			{
				std::lock_guard<std::mutex> lockGrayFrameGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::GrayFrame]);
				std::lock_guard<std::mutex> lockGrayFrameRGBGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::GrayFrameRGB]);

				gpuMats[GPUMatTypes::GrayFrame];
				gpuMats[GPUMatTypes::GrayFrameRGB];

				gpuMats.at(GPUMatTypes::GrayFrame).create(currentCamFrame.size(), CV_8UC1);
				gpuMats.at(GPUMatTypes::GrayFrameRGB).create(currentCamFrame.size(), currentCamFrame.type());


				break;
			}
			case FilterType::Sobel:
			{
				std::lock_guard<std::mutex> lockSobelFrameGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::SobelFrame]);
				std::lock_guard<std::mutex> lockSobelGradXGpuGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::SobelGradXGpu]);
				std::lock_guard<std::mutex> lockSobelGradYGpuGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::SobelGradYGpu]);

				gpuMats[GPUMatTypes::SobelFrame];
				gpuMats[GPUMatTypes::SobelGradXGpu];
				gpuMats[GPUMatTypes::SobelGradYGpu];

				gpuMats.at(GPUMatTypes::SobelFrame).create(currentCamFrame.size(), currentCamFrame.type());
				gpuMats.at(GPUMatTypes::SobelGradXGpu).create(currentCamFrame.size(), currentCamFrame.type());
				gpuMats.at(GPUMatTypes::SobelGradYGpu).create(currentCamFrame.size(), currentCamFrame.type());

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
			activeFiltersCount--;

			if (filteredFrames.empty())
			{
				std::lock_guard<std::mutex> lockCamFrameGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::CamFrame]);
				gpuMats.erase(GPUMatTypes::CamFrame);
			}

			switch (filterType)
			{
				case FilterType::Grayscale:
				{
					std::lock_guard<std::mutex> lockGrayFrameGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::GrayFrame]);
					std::lock_guard<std::mutex> lockGrayFrameRGBGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::GrayFrameRGB]);

					gpuMats.erase(GPUMatTypes::GrayFrame);
					gpuMats.erase(GPUMatTypes::GrayFrameRGB);
					break;
				}
				case FilterType::Sobel:
				{
					std::lock_guard<std::mutex> lockSobelFrameGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::SobelFrame]);
					std::lock_guard<std::mutex> lockSobelGradXGpuGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::SobelGradXGpu]);
					std::lock_guard<std::mutex> lockSobelGradYGpuGpuMatsMutexes(gpuMatsMutexes[GPUMatTypes::SobelGradYGpu]);

					gpuMats.erase(GPUMatTypes::SobelFrame);
					gpuMats.erase(GPUMatTypes::SobelGradXGpu);
					gpuMats.erase(GPUMatTypes::SobelGradYGpu);
					break;
				}
				default:
					break;
			}
		}
	}
}

void WebcamController::combinedFrameInitOrDestroy()
{
	if (combinedFiltersActive && combinedFiltersCount != 0)
	{
		cv::MatSize& currentCamFrameSize = currentCamFrame.size;
		currentFiltersCombinedFrame = cv::Mat(currentCamFrameSize().height, currentCamFrameSize().width * combinedFiltersCount, currentCamFrame.type());

		std::lock_guard<std::mutex> lockCurrentFiltersCombinedGpuMutex(gpuMatsMutexes[GPUMatTypes::CurrentFiltersCombined]);
		gpuMats[GPUMatTypes::CurrentFiltersCombined];
		gpuMats.at(GPUMatTypes::CurrentFiltersCombined).create(currentFiltersCombinedFrame.size(), currentFiltersCombinedFrame.type());
	}
	else if (combinedFiltersActive == false || combinedFiltersCount == 0)
	{
		currentFiltersCombinedFrame.release();

		std::lock_guard<std::mutex> lockCurrentFiltersCombinedGpuMutex(gpuMatsMutexes[GPUMatTypes::CurrentFiltersCombined]);
		gpuMats.erase(GPUMatTypes::CurrentFiltersCombined);
	}
}


void WebcamController::changedCombinedFiltersActive()
{
	combinedFrameInitOrDestroy();
}

void WebcamController::changedActiveCombinedFilters(FilterType filterType)
{
	if (combinedFilters.find(filterType)->second)
	{
		combinedFiltersCount++;
	}
	else
	{
		combinedFiltersCount--;
	}

	combinedFrameInitOrDestroy();
}

bool WebcamController::getFilteredMat(FilterType filterType, cv::Mat*& imageMat)
{
	const auto& filteredFrame = filteredFrames.find(filterType);
	if (filteredFrame != filteredFrames.end())
	{
		imageMat = &filteredFrame->second;
		return true;
	}

	return false;
}

const cv::Mat* WebcamController::getCurrentFiltersCombinedFrame()
{
	return &currentFiltersCombinedFrame;
}
