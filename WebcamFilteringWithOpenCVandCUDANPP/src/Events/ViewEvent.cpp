#include "ViewEvent.h"

ViewEvent::ViewEvent() :
	m_ViewEventType(ViewEventTypesEnum::None)
{
}

ViewEvent::ViewEvent(const ViewEventTypesEnum& eventType) :
	m_ViewEventType(eventType)
{
}

ViewEvent::~ViewEvent()
{
}

const ViewEventTypesEnum& ViewEvent::getViewEventType()
{
	return m_ViewEventType;
}
