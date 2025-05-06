#include "ViewEvent_ActivateCombinedFilter.h"

ViewEvent_ActivateCombinedFilter::ViewEvent_ActivateCombinedFilter() :
	ViewEvent(ViewEventTypesEnum::ActivateCombinedFilter)
{
	m_activateCombinedFilter = false;
}

void ViewEvent_ActivateCombinedFilter::setActivateCombinedFilter(const bool activateCombinedFilter)
{
	m_activateCombinedFilter = activateCombinedFilter;
}

bool ViewEvent_ActivateCombinedFilter::getActivateCombinedFilter()
{
	return m_activateCombinedFilter;
}
