#pragma once

#include "Events/ViewEvents/ViewChangeFilterEvents/ViewChangeFilterEvent.h"


class ChangeActiveFiltersOnCombinedFilter:
	public ViewChangeFilterEvent
{
public:
	ChangeActiveFiltersOnCombinedFilter();

	void setActiveFilterTypeOnCombined(const FilterTypeEnum& filterType, const bool isActive);
};
