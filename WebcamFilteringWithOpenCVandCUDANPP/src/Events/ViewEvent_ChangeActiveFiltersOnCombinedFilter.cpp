#include "ViewEvent_ChangeActiveFiltersOnCombinedFilter.h"

ViewEvent_ChangeActiveFiltersOnCombinedFilter::ViewEvent_ChangeActiveFiltersOnCombinedFilter() :
	ViewEvent(ViewEventTypesEnum::ChangeActiveFiltersOnCombinedFilter)
{
	m_filterType = FilterTypeEnum::None;
	m_isActive = false;
}

void ViewEvent_ChangeActiveFiltersOnCombinedFilter::setActiveFilterTypeOnCombined(const FilterTypeEnum& filterType, const bool isActive)
{
	m_filterType = filterType;
	m_isActive = isActive;
}

FilterTypeEnum ViewEvent_ChangeActiveFiltersOnCombinedFilter::getFilterType()
{
	return m_filterType;
}

bool ViewEvent_ChangeActiveFiltersOnCombinedFilter::getIsActive()
{
	return m_isActive;
}
