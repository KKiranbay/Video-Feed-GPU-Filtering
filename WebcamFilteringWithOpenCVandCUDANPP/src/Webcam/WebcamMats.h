#pragma once

#include <unordered_map>

#include <opencv4/opencv2/core/mat.hpp>

#include "Filters/FilterTypes.h"


class WebcamMats
{
public:
	WebcamMats() :
		activeMatsCount(0)
	{
		m_filteredMatsMap = {
			{ FilterTypeEnum::None, cv::Mat() },
			{ FilterTypeEnum::Grayscale, cv::Mat() },
			{ FilterTypeEnum::Sobel, cv::Mat() }
		};
	}

	void copyTo(WebcamMats& other)
	{
		other.activeMatsCount = activeMatsCount;

		for (auto& filterMat : m_filteredMatsMap)
		{
			other.m_filteredMatsMap.at(filterMat.first) = filterMat.second;
		}

		other.currentFiltersCombinedMat = currentFiltersCombinedMat;
	}

	int activeMatsCount;
	std::unordered_map<FilterTypeEnum, cv::Mat> m_filteredMatsMap;
	cv::Mat currentFiltersCombinedMat;
};