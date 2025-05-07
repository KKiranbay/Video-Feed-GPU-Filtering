#include "ActivateCombinedFilter.h"


ActivateCombinedFilter::ActivateCombinedFilter() :
	ViewEvent(ViewEventTypesEnum::ActivateCombinedFilter)
{
	m_activateCombinedFilter = false;
}

void ActivateCombinedFilter::setActivateCombinedFilter(const bool activateCombinedFilter)
{
	m_activateCombinedFilter = activateCombinedFilter;
}

bool ActivateCombinedFilter::getActivateCombinedFilter()
{
	return m_activateCombinedFilter;
}
