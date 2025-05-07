#include "ChangeActiveFiltersOnCombinedFilter.h"


ChangeActiveFiltersOnCombinedFilter::ChangeActiveFiltersOnCombinedFilter() :
	ViewChangeFilterEvent(ViewEventTypesEnum::ChangeActiveFiltersOnCombinedFilter)
{
	m_filterType = FilterTypeEnum::None;
	m_isActive = false;
}

void ChangeActiveFiltersOnCombinedFilter::setActiveFilterTypeOnCombined(const FilterTypeEnum& filterType, const bool isActive)
{
	m_filterType = filterType;
	m_isActive = isActive;
}
