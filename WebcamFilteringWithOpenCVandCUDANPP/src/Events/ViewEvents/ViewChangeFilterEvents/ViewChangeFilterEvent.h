#pragma once

#include "Events/ViewEvents/ViewEvent.h"
#include "Filters/FilterTypes.h"

class ViewChangeFilterEvent:
	public ViewEvent
{
public:
	FilterTypeEnum getFilterType();
	bool getIsActive();

protected:
	ViewChangeFilterEvent(const ViewEventTypesEnum& eventType);

	FilterTypeEnum m_filterType;
	bool m_isActive;
};

