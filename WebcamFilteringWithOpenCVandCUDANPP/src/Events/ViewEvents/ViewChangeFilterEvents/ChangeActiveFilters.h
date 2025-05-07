#pragma once

#include "Events/ViewEvents/ViewChangeFilterEvents/ViewChangeFilterEvent.h"
#include "Filters/FilterTypes.h"


class ChangeActiveFilters:
	public ViewChangeFilterEvent
{
public:
	ChangeActiveFilters();

	void setActiveFilterType(const FilterTypeEnum& filterType, const bool isActive);
};
