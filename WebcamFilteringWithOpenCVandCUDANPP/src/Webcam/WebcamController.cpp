#include "WebcamController.h"

#include <iostream>

#include <nppi_arithmetic_and_logical_operations.h>
#include <nppi_color_conversion.h>
#include <nppi_filtering_functions.h>

#include <opencv4/opencv2/cudaarithm.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>


WebcamController::WebcamController()
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

	activeFiltersStrings = {
		{ FilterTypeEnum::None, "None" },
		{ FilterTypeEnum::Grayscale, "Grayscale" },
		{ FilterTypeEnum::Sobel, "Sobel" }
	};

	combinedFiltersActive = false;
	combinedFiltersCount = 0;
	combinedFilters = {
		{ FilterTypeEnum::None, false },
		{ FilterTypeEnum::Grayscale, false },
		{ FilterTypeEnum::Sobel, false }
	};

	videoCaptureCanBeStarted = false;

	initFilteredMatsAndMutexesMap();
	initGpuMatsAndMutexesMap();
}

void WebcamController::initFilteredMatsAndMutexesMap()
{
	std::array<FilterTypeEnum, 3> filterTypes = {
		FilterTypeEnum::None,
		FilterTypeEnum::Grayscale,
		FilterTypeEnum::Sobel
	};

	for (auto& filterType : filterTypes)
	{
		filteredMatsAndMutexesMap[filterType];
	}
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
		gpuMatsAndMutexesMap[gpuMatType];
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

		if (filteredMatsAndMutexesMap.empty())
		{
			continue;
		}

		GpuMatAndMutex& camFrameGpuMatAndMutex = gpuMatsAndMutexesMap.at(GPUMatTypesEnum::CamFrame);
		cv::cuda::GpuMat& camFrameGpuMat = camFrameGpuMatAndMutex.gpuMat;

		{
			std::lock_guard<std::mutex> lockCamFrameGpuMat(camFrameGpuMatAndMutex.gpuMutex);

			camFrameGpuMat.upload(currentCamFrame);
			cv::cuda::flip(camFrameGpuMat, camFrameGpuMat, 1);
		}

		for (const auto& filter : activeFiltersMap)
		{
			if (filter.second == false)
				continue;

			switch (filter.first)
			{
			case FilterTypeEnum::None:
			{
				generateCameraFrame();
				break;
			}
			case FilterTypeEnum::Grayscale:
			{
				generateGrayscaleRGBFrame();
				break;
			}
			case FilterTypeEnum::Sobel:
			{
				generateSobelFilteredFrame();
				break;
			}
			default:
				break;
			}
		}

		std::lock_guard<std::mutex> lockCombinedFiltersCount(combinedFiltersCountMutex);
		if (combinedFiltersActive
			&& combinedFiltersCount != 0
			&& gpuMatsAndMutexesMap.find(GPUMatTypesEnum::CurrentFiltersCombined) != gpuMatsAndMutexesMap.end())
		{
			generateCombinedFilteredFrame();
		}
	}
}

// Generate function is used by the thread for capturing frames
void WebcamController::generateCameraFrame()
{
	std::lock_guard<std::mutex> lockFilterMutexes(filteredMatsAndMutexesMap.at(FilterTypeEnum::None).matMutex);

	std::lock_guard<std::mutex> lockCamFrameGpuMat(gpuMatsAndMutexesMap.at(GPUMatTypesEnum::CamFrame).gpuMutex);
	gpuMatsAndMutexesMap.at(GPUMatTypesEnum::CamFrame).gpuMat.download(filteredMatsAndMutexesMap.at(FilterTypeEnum::None).mat);
}

// Generate function is used by the thread for capturing frames
void WebcamController::generateGrayscaleRGBFrame()
{
	cv::cuda::GpuMat& camFrameGpuMat = gpuMatsAndMutexesMap.at(GPUMatTypesEnum::CamFrame).gpuMat;

	GpuMatAndMutex& grayFrameGpuMatAndMutex = gpuMatsAndMutexesMap.at(GPUMatTypesEnum::GrayFrame);
	cv::cuda::GpuMat& grayFrameGpuMat = grayFrameGpuMatAndMutex.gpuMat;
	std::mutex& grayFrameGpuMutex = grayFrameGpuMatAndMutex.gpuMutex;

	grayFrameGpuMutex.lock();

	{
		std::lock_guard<std::mutex> lockCamFrameGpuMat(gpuMatsAndMutexesMap.at(GPUMatTypesEnum::CamFrame).gpuMutex);
		NppStatus status = nppiRGBToGray_8u_C3C1R(camFrameGpuMat.ptr(), static_cast<int>(camFrameGpuMat.step),
			grayFrameGpuMat.ptr(), static_cast<int>(grayFrameGpuMat.step),
			{ camFrameGpuMat.cols, camFrameGpuMat.rows });

		if (status != NPP_SUCCESS)
		{
			grayFrameGpuMutex.unlock();

			std::cerr << "Error computing grayscale gradient: " << status << std::endl;
			return;
		}
	}

	GpuMatAndMutex& grayFrameRGBGpuMatAndMutex = gpuMatsAndMutexesMap.at(GPUMatTypesEnum::GrayFrameRGB);
	cv::cuda::GpuMat& grayFrameRGBGpuMat = grayFrameRGBGpuMatAndMutex.gpuMat;
	std::mutex& grayFrameRGBGpuMutex = grayFrameRGBGpuMatAndMutex.gpuMutex;

	grayFrameRGBGpuMutex.lock();

	cv::cuda::cvtColor(grayFrameGpuMat, grayFrameRGBGpuMat, cv::COLOR_GRAY2RGB);

	grayFrameGpuMutex.unlock();

	MatAndMutex& grayscaleFrameMatAndMutex = filteredMatsAndMutexesMap.at(FilterTypeEnum::Grayscale);

	std::lock_guard<std::mutex> lockFilterMutexes(grayscaleFrameMatAndMutex.matMutex);

	grayFrameRGBGpuMat.download(grayscaleFrameMatAndMutex.mat);

	grayFrameRGBGpuMutex.unlock();
}

// Generate function is used by the thread for capturing frames
void WebcamController::generateSobelFilteredFrame()
{
	GpuMatAndMutex& camFrameGpuMatAndMutex = gpuMatsAndMutexesMap.at(GPUMatTypesEnum::CamFrame);
	cv::cuda::GpuMat& camFrameGpuMat = camFrameGpuMatAndMutex.gpuMat;

	GpuMatAndMutex& sobelGradXGpuMatAndMutex = gpuMatsAndMutexesMap.at(GPUMatTypesEnum::SobelGradXGpu);

	const Npp8u* camFrameGpuMatPtr = static_cast<const Npp8u*>(camFrameGpuMat.ptr());

	cv::cuda::GpuMat& gradXGpuMat = sobelGradXGpuMatAndMutex.gpuMat;
	Npp8u* gradXGpuPtr = static_cast<Npp8u*>(gradXGpuMat.ptr());
	NppiSize roi = { camFrameGpuMat.cols, camFrameGpuMat.rows };

	std::mutex& camFrameGpuMutex = camFrameGpuMatAndMutex.gpuMutex;
	camFrameGpuMutex.lock();

	std::mutex& sobelGradXGpuMutex = sobelGradXGpuMatAndMutex.gpuMutex;
	sobelGradXGpuMutex.lock();

	NppStatus status = nppiFilterSobelHoriz_8u_C3R(camFrameGpuMatPtr, static_cast<Npp32s>(camFrameGpuMat.step),
		gradXGpuPtr, static_cast<Npp32s>(gradXGpuMat.step),
		roi);
	if (status != NPP_SUCCESS)
	{
		camFrameGpuMutex.unlock();
		sobelGradXGpuMutex.unlock();

		std::cerr << "Error computing horizontal gradient: " << status << std::endl;
		return;
	}

	GpuMatAndMutex& sobelGradYGpuMatAndMutex = gpuMatsAndMutexesMap.at(GPUMatTypesEnum::SobelGradYGpu);
	cv::cuda::GpuMat& gradYGpuMat = sobelGradYGpuMatAndMutex.gpuMat;
	Npp8u* gradYGpuPtr = static_cast<Npp8u*>(gradYGpuMat.ptr());

	std::mutex& sobelGradYGpuMutex = sobelGradYGpuMatAndMutex.gpuMutex;
	sobelGradYGpuMutex.lock();

	status = nppiFilterSobelVert_8u_C3R(camFrameGpuMatPtr, static_cast<Npp32s>(camFrameGpuMat.step),
		gradYGpuPtr, static_cast<Npp32s>(gradYGpuMat.step),
		roi);

	camFrameGpuMutex.unlock();

	if (status != NPP_SUCCESS)
	{
		sobelGradXGpuMutex.unlock();
		sobelGradYGpuMutex.unlock();

		std::cerr << "Error computing vertical gradient: " << status << std::endl;
		return;
	}

	GpuMatAndMutex& sobelFrameGpuMatAndMutex = gpuMatsAndMutexesMap.at(GPUMatTypesEnum::SobelFrame);
	cv::cuda::GpuMat& sobelFrameGpuMat = sobelFrameGpuMatAndMutex.gpuMat;
	std::mutex& sobelFrameGpuMutex = sobelFrameGpuMatAndMutex.gpuMutex;

	sobelFrameGpuMutex.lock();

	status = nppiAdd_8u_C3RSfs(gradXGpuPtr, static_cast<int>(gradXGpuMat.step),
		gradYGpuPtr, static_cast<int>(gradYGpuMat.step),
		static_cast<Npp8u*>(sobelFrameGpuMat.ptr()), static_cast<int>(sobelFrameGpuMat.step),
		roi, 0); // no scaling

	sobelGradXGpuMutex.unlock();
	sobelGradYGpuMutex.unlock();

	if (status != NPP_SUCCESS)
	{
		sobelFrameGpuMutex.unlock();

		std::cerr << "Error computing magnitude: " << status << std::endl;
		return;
	}

	MatAndMutex& sobelFrameMatAndMutex = filteredMatsAndMutexesMap.at(FilterTypeEnum::Sobel);

	std::lock_guard<std::mutex> lockSobelFrameMutex(sobelFrameMatAndMutex.matMutex);
	sobelFrameGpuMat.download(sobelFrameMatAndMutex.mat);

	sobelFrameGpuMutex.unlock();
}

// Generate function is used by the thread for capturing frames
void WebcamController::generateCombinedFilteredFrame()
{
	GpuMatAndMutex& currentFiltersCombinedGpuMatAndMutex = gpuMatsAndMutexesMap.at(GPUMatTypesEnum::CurrentFiltersCombined);

	std::mutex& currentFiltersCombinedGpuMutex = currentFiltersCombinedGpuMatAndMutex.gpuMutex;

	std::lock_guard<std::mutex> lockGpuMatMutex(currentFiltersCombinedGpuMutex);

	cv::cuda::GpuMat& currentFiltersCombinedFrameGpuMat = currentFiltersCombinedGpuMatAndMutex.gpuMat;
	int combinedFiltersPlace = 0;

	for (const auto& filter : combinedFilters)
	{
		if (filter.second == false)
			continue;

		GpuMatAndMutex* gpuMatAndMutex = nullptr;

		switch (filter.first)
		{
		case FilterTypeEnum::None:
		{
			gpuMatAndMutex = &gpuMatsAndMutexesMap.at(GPUMatTypesEnum::CamFrame);
			break;
		}
		case FilterTypeEnum::Grayscale:
		{
			gpuMatAndMutex = &gpuMatsAndMutexesMap.at(GPUMatTypesEnum::GrayFrameRGB);
			break;
		}
		case FilterTypeEnum::Sobel:
		{
			gpuMatAndMutex = &gpuMatsAndMutexesMap.at(GPUMatTypesEnum::SobelFrame);
			break;
		}
		default:
			break;
		}

		if (gpuMatAndMutex == nullptr)
			continue;

		cv::cuda::GpuMat& gpuMat = gpuMatAndMutex->gpuMat;
		std::mutex& gpuMatMutex = gpuMatAndMutex->gpuMutex;

		std::lock_guard<std::mutex> lockGpuMatMutex(gpuMatMutex);

		int gpuMatHeight = gpuMat.size().height;
		int gpuMatWidth = gpuMat.size().width;

		gpuMat.copyTo(currentFiltersCombinedFrameGpuMat
			.rowRange(0, gpuMatHeight)
			.colRange(gpuMatWidth * combinedFiltersPlace, gpuMatWidth * (combinedFiltersPlace + 1)));

		combinedFiltersPlace++;
	}

	std::lock_guard<std::mutex> lockCurrentFiltersCombinedFrameMutex(currentFiltersCombinedMatAndMutex.matMutex);
	currentFiltersCombinedFrameGpuMat.download(currentFiltersCombinedMatAndMutex.mat);
}

// This function is used by only view
void WebcamController::changedActiveFilter(FilterTypeEnum FilterTypeEnum)
{
	bool& active = activeFiltersMap.at(FilterTypeEnum);

	if (active)
	{
		activeFiltersCount++;
		filteredMatsAndMutexesMap.at(FilterTypeEnum).mat.create(currentCamFrame.size(), currentCamFrame.type());

		if (activeFiltersCount == 1)
		{
			gpuMatsAndMutexesMap.at(GPUMatTypesEnum::CamFrame).gpuMat.create(currentCamFrame.size(), currentCamFrame.type());
		}

		switch (FilterTypeEnum)
		{
		case FilterTypeEnum::Grayscale:
		{
			gpuMatsAndMutexesMap.at(GPUMatTypesEnum::GrayFrame).gpuMat.create(currentCamFrame.size(), CV_8UC1);
			gpuMatsAndMutexesMap.at(GPUMatTypesEnum::GrayFrameRGB).gpuMat.create(currentCamFrame.size(), currentCamFrame.type());

			break;
		}
		case FilterTypeEnum::Sobel:
		{
			gpuMatsAndMutexesMap.at(GPUMatTypesEnum::SobelFrame).gpuMat.create(currentCamFrame.size(), currentCamFrame.type());
			gpuMatsAndMutexesMap.at(GPUMatTypesEnum::SobelGradXGpu).gpuMat.create(currentCamFrame.size(), currentCamFrame.type());
			gpuMatsAndMutexesMap.at(GPUMatTypesEnum::SobelGradYGpu).gpuMat.create(currentCamFrame.size(), currentCamFrame.type());

			break;
		}
		default:
			break;
		}
	}
	else
	{
		activeFiltersCount--;

		filteredMatsAndMutexesMap.at(FilterTypeEnum).mat.release();

		switch (FilterTypeEnum)
		{
		case FilterTypeEnum::Grayscale:
		{
			gpuMatsAndMutexesMap.at(GPUMatTypesEnum::GrayFrame).gpuMat.release();
			gpuMatsAndMutexesMap.at(GPUMatTypesEnum::GrayFrameRGB).gpuMat.release();
			break;
		}
		case FilterTypeEnum::Sobel:
		{
			gpuMatsAndMutexesMap.at(GPUMatTypesEnum::SobelFrame).gpuMat.release();
			gpuMatsAndMutexesMap.at(GPUMatTypesEnum::SobelGradXGpu).gpuMat.release();
			gpuMatsAndMutexesMap.at(GPUMatTypesEnum::SobelGradYGpu).gpuMat.release();
			break;
		}
		default:
			break;
		}
	}
}

// This function is used by only view
void WebcamController::combinedFrameInitOrDestroy()
{
	std::lock_guard<std::mutex> lockCurrentFiltersCombinedFrameMatMutex(currentFiltersCombinedMatAndMutex.matMutex);

	GpuMatAndMutex& currentFiltersCombinedGpuMatAndMutex = gpuMatsAndMutexesMap.at(GPUMatTypesEnum::CurrentFiltersCombined);
	std::lock_guard<std::mutex> lockCurrentFiltersCombinedGpuMatMutex(currentFiltersCombinedGpuMatAndMutex.gpuMutex);

	if (combinedFiltersActive && combinedFiltersCount != 0)
	{
		cv::MatSize& currentCamFrameSize = currentCamFrame.size;
		currentFiltersCombinedMatAndMutex.mat.create(currentCamFrameSize().height, currentCamFrameSize().width * combinedFiltersCount, currentCamFrame.type());

		currentFiltersCombinedGpuMatAndMutex.gpuMat.create(currentFiltersCombinedMatAndMutex.mat.size(), currentFiltersCombinedMatAndMutex.mat.type());
	}
	else if (combinedFiltersActive == false || combinedFiltersCount == 0)
	{
		currentFiltersCombinedMatAndMutex.mat.release();

		currentFiltersCombinedGpuMatAndMutex.gpuMat.release();
	}
}

// This function is used by only view
void WebcamController::changedCombinedFiltersActive()
{
	combinedFrameInitOrDestroy();
}

// This function is used by only view
void WebcamController::changedActiveCombinedFilters(FilterTypeEnum FilterTypeEnum)
{
	std::lock_guard<std::mutex> lockCombinedFiltersCount(combinedFiltersCountMutex);

	if (combinedFilters.find(FilterTypeEnum)->second)
	{
		combinedFiltersCount++;
	}
	else
	{
		combinedFiltersCount--;
	}

	combinedFrameInitOrDestroy();
}

// This function is used by only view
const cv::Mat* WebcamController::getFilteredMat(FilterTypeEnum filterType)
{
	MatAndMutex& matAndMutex = filteredMatsAndMutexesMap.at(filterType);
	return &matAndMutex.mat;
}

// This function is used by only view
const cv::Mat* WebcamController::getCurrentFiltersCombinedFrame()
{
	return &currentFiltersCombinedMatAndMutex.mat;
}
