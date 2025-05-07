#include "ChangeActiveFilters.h"

ChangeActiveFilters::ChangeActiveFilters() :
	ViewChangeFilterEvent(ViewEventTypesEnum::ChangeActiveFilters)
{
	m_filterType = FilterTypeEnum::None;
	m_isActive = false;
}

void ChangeActiveFilters::setActiveFilterType(const FilterTypeEnum& filterType, const bool isActive)
{
	m_filterType = filterType;
	m_isActive = isActive;
}
