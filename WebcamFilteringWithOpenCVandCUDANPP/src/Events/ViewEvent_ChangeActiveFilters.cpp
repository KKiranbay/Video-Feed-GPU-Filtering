#include "ViewEvent_ChangeActiveFilters.h"

ViewEvent_ChangeActiveFilters::ViewEvent_ChangeActiveFilters() :
	ViewEvent(ViewEventTypesEnum::ChangeActiveFilters)
{
	m_filterType = FilterTypeEnum::None;
	m_isActive = false;
}

void ViewEvent_ChangeActiveFilters::setActiveFilterType(const FilterTypeEnum& filterType, const bool isActive)
{
	m_filterType = filterType;
	m_isActive = isActive;
}

FilterTypeEnum ViewEvent_ChangeActiveFilters::getFilterType()
{
	return m_filterType;
}

bool ViewEvent_ChangeActiveFilters::getIsActive()
{
	return m_isActive;
}
