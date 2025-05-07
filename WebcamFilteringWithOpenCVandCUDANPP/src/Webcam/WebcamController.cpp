#include "WebcamController.h"

#include <iostream>

#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_color_conversion.h>
#include <nppi_filtering_functions.h>

#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>

#include "Events/ViewEvents/ActivateCombinedFilter.h"
#include "Events/ViewEvents/ViewChangeFilterEvents/ChangeActiveFilters.h"
#include "Events/ViewEvents/ViewChangeFilterEvents/ChangeActiveFiltersOnCombinedFilter.h"

#include "WebcamView.h"



WebcamController::WebcamController(WebcamView* parentView) :
	parentView(parentView)
{
	initVariables();
	initVideoCapture();
}

void WebcamController::initVariables()
{
	activeFiltersCount = 0;
	activeFiltersMap = {
		{ FilterTypeEnum::None, false },
		{ FilterTypeEnum::Grayscale, false },
		{ FilterTypeEnum::Sobel, false }
	};

	combinedFiltersActive = false;
	combinedFiltersCount = 0;
	combinedFilters = {
		{ FilterTypeEnum::None, false },
		{ FilterTypeEnum::Grayscale, false },
		{ FilterTypeEnum::Sobel, false }
	};

	videoCaptureCanBeStarted = false;

	initGpuMatsAndMutexesMap();
}

void WebcamController::initGpuMatsAndMutexesMap()
{
	std::array<GPUMatTypesEnum, 7> gpuMatTypes = {
		GPUMatTypesEnum::CamFrame,
		GPUMatTypesEnum::GrayFrame,
		GPUMatTypesEnum::GrayFrameRGB,
		GPUMatTypesEnum::SobelFrame,
		GPUMatTypesEnum::SobelGradXGpu,
		GPUMatTypesEnum::SobelGradYGpu,
		GPUMatTypesEnum::CurrentFiltersCombined
	};

	for (auto& gpuMatType : gpuMatTypes)
	{
		gpuMatsMap[gpuMatType];
	}
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

// This function is used by only view
void WebcamController::startVideoCapture()
{
	if (videoCaptureCanBeStarted)
		videoCaptureThread = std::jthread(&WebcamController::startVideoCaptureThread, this);
}

// Thread function for capturing frames
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

		processEvents();

		if (activeFiltersCount == 0)
			continue;

		flipCameraFrame();

		generateActiveFilters();
	}
}

void WebcamController::processEvents()
{
	std::shared_ptr<ViewEvent> viewEvent;
	while ((viewEvent = parentView->getEventFromQueue()) != nullptr)
	{
		switch (viewEvent->getViewEventType())
		{
			case ViewEventTypesEnum::ActivateCombinedFilter:
				processChangedCombinedFiltersActive(viewEvent);
				break;
			case ViewEventTypesEnum::ChangeActiveFilters:
				processChangedActiveFilters(viewEvent);
				break;
			case ViewEventTypesEnum::ChangeActiveFiltersOnCombinedFilter:
				processChangedActiveFiltersOnCombinedFilters(viewEvent);
				break;
		}
	}
}

void WebcamController::processChangedCombinedFiltersActive(std::shared_ptr<ViewEvent> event)
{
	combinedFiltersActive = std::static_pointer_cast<ActivateCombinedFilter>(event)->getActivateCombinedFilter();

	combinedFrameInitOrDestroy();
}

void WebcamController::processChangedActiveFilters(std::shared_ptr<ViewEvent> event)
{
	std::shared_ptr<ChangeActiveFilters> changeActiveFiltersEventPtr = std::static_pointer_cast<ChangeActiveFilters>(event);

	FilterTypeEnum filterType = changeActiveFiltersEventPtr->getFilterType();
	bool isActive = changeActiveFiltersEventPtr->getIsActive();
	bool& refActive = activeFiltersMap.at(filterType);

	if (refActive == isActive)
		return;

	if (refActive = isActive)
	{
		activeFiltersCount++;

		if (activeFiltersCount == 1)
		{
			gpuMatsMap.at(GPUMatTypesEnum::CamFrame).create(currentCamFrame.size(), currentCamFrame.type());
		}

		switch (filterType)
		{
			case FilterTypeEnum::Grayscale:
			{
				gpuMatsMap.at(GPUMatTypesEnum::GrayFrame).create(currentCamFrame.size(), CV_8UC1);
				gpuMatsMap.at(GPUMatTypesEnum::GrayFrameRGB).create(currentCamFrame.size(), currentCamFrame.type());
				break;
			}
			case FilterTypeEnum::Sobel:
			{
				gpuMatsMap.at(GPUMatTypesEnum::SobelFrame).create(currentCamFrame.size(), currentCamFrame.type());
				gpuMatsMap.at(GPUMatTypesEnum::SobelGradXGpu).create(currentCamFrame.size(), currentCamFrame.type());
				gpuMatsMap.at(GPUMatTypesEnum::SobelGradYGpu).create(currentCamFrame.size(), currentCamFrame.type());
				break;
			}
			default:
				break;
		}
	}
	else
	{
		{
			std::lock_guard<std::mutex> lock(m_WebcamMatsMutex);
			m_ControllersWebcamMats.m_filteredMatsMap.at(filterType).release();
			m_ControllersWebcamMats.activeMatsCount--;
		}

		activeFiltersCount--;

		switch (filterType)
		{
			case FilterTypeEnum::Grayscale:
			{
				gpuMatsMap.at(GPUMatTypesEnum::GrayFrame).release();
				gpuMatsMap.at(GPUMatTypesEnum::GrayFrameRGB).release();
				break;
			}
			case FilterTypeEnum::Sobel:
			{
				gpuMatsMap.at(GPUMatTypesEnum::SobelFrame).release();
				gpuMatsMap.at(GPUMatTypesEnum::SobelGradXGpu).release();
				gpuMatsMap.at(GPUMatTypesEnum::SobelGradYGpu).release();
				break;
			}
			default:
				break;
		}

		changeActiveCombinedFilters(filterType, false);
	}
}

void WebcamController::processChangedActiveFiltersOnCombinedFilters(std::shared_ptr<ViewEvent> event)
{
	std::shared_ptr<ChangeActiveFiltersOnCombinedFilter> changeActiveFiltersOnCombinedFilterEventPtr = std::static_pointer_cast<ChangeActiveFiltersOnCombinedFilter>(event);

	changeActiveCombinedFilters(changeActiveFiltersOnCombinedFilterEventPtr->getFilterType(), changeActiveFiltersOnCombinedFilterEventPtr->getIsActive());
}

void WebcamController::changeActiveCombinedFilters(FilterTypeEnum filterType, bool isActive)
{
	bool& refCombined = combinedFilters.at(filterType);

	if (refCombined == isActive)
		return;

	if (refCombined = isActive)
	{
		combinedFiltersCount++;
	}
	else
	{
		combinedFiltersCount--;
	}

	combinedFrameInitOrDestroy();
}

void WebcamController::combinedFrameInitOrDestroy()
{
	cv::cuda::GpuMat& currentFiltersCombinedGpuMat = gpuMatsMap.at(GPUMatTypesEnum::CurrentFiltersCombined);

	if (combinedFiltersActive && combinedFiltersCount != 0)
	{
		cv::MatSize& currentCamFrameSize = currentCamFrame.size;
		currentFiltersCombinedGpuMat.create(currentCamFrameSize().height, currentCamFrameSize().width * combinedFiltersCount, currentCamFrame.type());

		std::lock_guard<std::mutex> lock(m_WebcamMatsMutex);
		m_ControllersWebcamMats.currentFiltersCombinedMat.create(currentCamFrameSize().height, currentCamFrameSize().width * combinedFiltersCount, currentCamFrame.type());
	}
	else if (combinedFiltersActive == false || combinedFiltersCount == 0)
	{
		currentFiltersCombinedGpuMat.release();

		std::lock_guard<std::mutex> lock(m_WebcamMatsMutex);
		m_ControllersWebcamMats.currentFiltersCombinedMat.release();
	}
}

void WebcamController::flipCameraFrame()
{
	cv::cuda::GpuMat& camFrameGpuMat = gpuMatsMap.at(GPUMatTypesEnum::CamFrame);
	camFrameGpuMat.upload(currentCamFrame);
	cv::cuda::flip(camFrameGpuMat, camFrameGpuMat, 1);
}

void WebcamController::generateActiveFilters()
{
	std::vector<std::thread> generateFramesThreads(activeFiltersCount);

	for (const auto& filter : activeFiltersMap)
	{
		if (filter.second == false)
			continue;

		switch (filter.first)
		{
			case FilterTypeEnum::None:
			{
				generateFramesThreads.emplace_back(&WebcamController::generateCameraFrame, this);
				break;
			}
			case FilterTypeEnum::Grayscale:
			{
				generateFramesThreads.emplace_back(&WebcamController::generateGrayscaleRGBFrame, this);
				break;
			}
			case FilterTypeEnum::Sobel:
			{
				generateFramesThreads.emplace_back(&WebcamController::generateSobelFilteredFrame, this);
				break;
			}
			default:
				break;
		}
	}

	for (auto& thread : generateFramesThreads)
	{
		if (thread.joinable())
			thread.join();
	}

	if (combinedFiltersActive
		&& combinedFiltersCount != 0
		&& gpuMatsMap.find(GPUMatTypesEnum::CurrentFiltersCombined) != gpuMatsMap.end())
	{
		generateCombinedFilteredFrame();
	}
}

// Generate function is used by the thread for capturing frames
void WebcamController::generateCameraFrame()
{
	cv::cuda::GpuMat& camFrameGpuMat = gpuMatsMap.at(GPUMatTypesEnum::CamFrame);
	cv::Mat& webcamMat = m_ControllersWebcamMats.m_filteredMatsMap.at(FilterTypeEnum::None);

	std::lock_guard<std::mutex> lock(m_WebcamMatsMutex);

	if (webcamMat.empty())
		m_ControllersWebcamMats.activeMatsCount++;

	camFrameGpuMat.download(webcamMat);
}

// Generate function is used by the thread for capturing frames
void WebcamController::generateGrayscaleRGBFrame()
{
	cv::cuda::GpuMat& camFrameGpuMat = gpuMatsMap.at(GPUMatTypesEnum::CamFrame);

	cv::cuda::GpuMat& grayFrameGpuMat = gpuMatsMap.at(GPUMatTypesEnum::GrayFrame);

	NppStatus status = nppiRGBToGray_8u_C3C1R(camFrameGpuMat.ptr(), static_cast<int>(camFrameGpuMat.step),
											  grayFrameGpuMat.ptr(), static_cast<int>(grayFrameGpuMat.step),
											  { camFrameGpuMat.cols, camFrameGpuMat.rows });

	if (status != NPP_SUCCESS)
	{
		std::cerr << "Error computing grayscale gradient: " << status << std::endl;
		return;
	}

	cv::cuda::GpuMat& grayFrameRGBGpuMat = gpuMatsMap.at(GPUMatTypesEnum::GrayFrameRGB);
	cv::cuda::cvtColor(grayFrameGpuMat, grayFrameRGBGpuMat, cv::COLOR_GRAY2RGB);

	cv::Mat& webcamMat = m_ControllersWebcamMats.m_filteredMatsMap.at(FilterTypeEnum::Grayscale);

	std::lock_guard<std::mutex> lock(m_WebcamMatsMutex);

	if (webcamMat.empty())
		m_ControllersWebcamMats.activeMatsCount++;

	grayFrameRGBGpuMat.download(webcamMat);
}

// Generate function is used by the thread for capturing frames
void WebcamController::generateSobelFilteredFrame()
{
	cv::cuda::GpuMat& camFrameGpuMat = gpuMatsMap.at(GPUMatTypesEnum::CamFrame);

	NppiSize roi = { camFrameGpuMat.cols, camFrameGpuMat.rows };
	const Npp8u* camFrameGpuMatPtr = static_cast<const Npp8u*>(camFrameGpuMat.ptr());

	cv::cuda::GpuMat& gradXGpuMat = gpuMatsMap.at(GPUMatTypesEnum::SobelGradXGpu);
	Npp8u* gradXGpuPtr = static_cast<Npp8u*>(gradXGpuMat.ptr());

	NppStatus status = nppiFilterSobelHoriz_8u_C3R(camFrameGpuMatPtr, static_cast<Npp32s>(camFrameGpuMat.step),
												   gradXGpuPtr, static_cast<Npp32s>(gradXGpuMat.step),
												   roi);
	if (status != NPP_SUCCESS)
	{
		std::cerr << "Error computing horizontal gradient: " << status << std::endl;
		return;
	}

	cv::cuda::GpuMat& gradYGpuMat = gpuMatsMap.at(GPUMatTypesEnum::SobelGradYGpu);
	Npp8u* gradYGpuPtr = static_cast<Npp8u*>(gradYGpuMat.ptr());

	status = nppiFilterSobelVert_8u_C3R(camFrameGpuMatPtr, static_cast<Npp32s>(camFrameGpuMat.step),
										gradYGpuPtr, static_cast<Npp32s>(gradYGpuMat.step),
										roi);
	if (status != NPP_SUCCESS)
	{
		std::cerr << "Error computing vertical gradient: " << status << std::endl;
		return;
	}

	cv::cuda::GpuMat& sobelFrameGpuMat = gpuMatsMap.at(GPUMatTypesEnum::SobelFrame);

	status = nppiAdd_8u_C3RSfs(gradXGpuPtr, static_cast<int>(gradXGpuMat.step),
							   gradYGpuPtr, static_cast<int>(gradYGpuMat.step),
							   static_cast<Npp8u*>(sobelFrameGpuMat.ptr()), static_cast<int>(sobelFrameGpuMat.step),
							   roi, 0); // no scaling
	if (status != NPP_SUCCESS)
	{
		std::cerr << "Error computing magnitude: " << status << std::endl;
		return;
	}

	cv::Mat& webcamMat = m_ControllersWebcamMats.m_filteredMatsMap.at(FilterTypeEnum::Sobel);

	std::lock_guard<std::mutex> lock(m_WebcamMatsMutex);

	if (webcamMat.empty())
		m_ControllersWebcamMats.activeMatsCount++;

	sobelFrameGpuMat.download(webcamMat);
}

// Generate function is used by the thread for capturing frames
void WebcamController::generateCombinedFilteredFrame()
{
	cv::cuda::GpuMat& currentFiltersCombinedFrameGpuMat = gpuMatsMap.at(GPUMatTypesEnum::CurrentFiltersCombined);
	int combinedFiltersPlace = 0;

	for (const auto& filter : combinedFilters)
	{
		if (filter.second == false)
			continue;

		cv::cuda::GpuMat* gpuMatPtr = nullptr;

		switch (filter.first)
		{
			case FilterTypeEnum::None:
			{
				gpuMatPtr = &gpuMatsMap.at(GPUMatTypesEnum::CamFrame);
				break;
			}
			case FilterTypeEnum::Grayscale:
			{
				gpuMatPtr = &gpuMatsMap.at(GPUMatTypesEnum::GrayFrameRGB);
				break;
			}
			case FilterTypeEnum::Sobel:
			{
				gpuMatPtr = &gpuMatsMap.at(GPUMatTypesEnum::SobelFrame);
				break;
			}
			default:
				break;
		}

		if (gpuMatPtr == nullptr)
			continue;

		cv::cuda::GpuMat& gpuMat = *gpuMatPtr;

		int gpuMatHeight = gpuMat.size().height;
		int gpuMatWidth = gpuMat.size().width;

		gpuMat.copyTo(currentFiltersCombinedFrameGpuMat
					  .rowRange(0, gpuMatHeight)
					  .colRange(gpuMatWidth * combinedFiltersPlace, gpuMatWidth * (combinedFiltersPlace + 1)));

		combinedFiltersPlace++;
	}

	std::lock_guard<std::mutex> lock(m_WebcamMatsMutex);
	currentFiltersCombinedFrameGpuMat.download(m_ControllersWebcamMats.currentFiltersCombinedMat);
}

void WebcamController::getMats(WebcamMats& webcamMatsFromView)
{
	std::lock_guard<std::mutex> lock(m_WebcamMatsMutex);

	m_ControllersWebcamMats.copyTo(webcamMatsFromView);
}
