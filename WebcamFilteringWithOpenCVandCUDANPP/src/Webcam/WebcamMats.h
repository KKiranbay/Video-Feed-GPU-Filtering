#pragma once

#include <unordered_map>

#include <opencv4/opencv2/core/mat.hpp>

#include "Filters/FilterTypes.h"


class WebcamMats
{
public:
	WebcamMats()
	{
		activeMatsCount = 0;
		filteredMatsMap = {
			{ FilterTypeEnum::None, cv::Mat() },
			{ FilterTypeEnum::Grayscale, cv::Mat() },
			{ FilterTypeEnum::Sobel, cv::Mat() }
		};
	}

	int activeMatsCount;
	std::unordered_map<FilterTypeEnum, cv::Mat> filteredMatsMap;
	cv::Mat currentFiltersCombinedMat;
};