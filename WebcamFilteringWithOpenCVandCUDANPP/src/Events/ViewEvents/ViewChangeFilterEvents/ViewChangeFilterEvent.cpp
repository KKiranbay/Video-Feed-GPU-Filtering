#include "ViewChangeFilterEvent.h"


ViewChangeFilterEvent::ViewChangeFilterEvent(const ViewEventTypesEnum& eventType) :
	ViewEvent(eventType)
{
}

FilterTypeEnum ViewChangeFilterEvent::getFilterType()
{
	return m_filterType;
}

bool ViewChangeFilterEvent::getIsActive()
{
	return m_isActive;
}


