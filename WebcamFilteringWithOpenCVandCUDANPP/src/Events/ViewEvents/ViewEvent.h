#pragma once

#include "Events/ViewEvents/ViewEventTypes.h"


class ViewEvent
{
public:
	virtual ~ViewEvent();

	const ViewEventTypesEnum& getViewEventType();

protected:
	ViewEvent(const ViewEventTypesEnum& eventType);

	ViewEventTypesEnum m_ViewEventType;
};
