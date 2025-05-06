#pragma once

#include "Filters/FilterTypes.h"
#include "ViewEvent.h"


class ViewEvent_ChangeActiveFilters:
	public ViewEvent
{
public:
	ViewEvent_ChangeActiveFilters();

	void setActiveFilterType(const FilterTypeEnum& filterType, const bool isActive);

	FilterTypeEnum getFilterType();
	bool getIsActive();

private:
	FilterTypeEnum m_filterType;
	bool m_isActive;
};
