#include "ViewEventQueue.h"

std::shared_ptr<ViewEvent> ViewEventQueue::popViewEvent()
{
	std::unique_lock lock(m_ViewEventMutex);

	if (m_ViewEventQueue.empty())
	{
		return nullptr;
	}

	std::shared_ptr<ViewEvent> frontViewEvent = m_ViewEventQueue.front();
	m_ViewEventQueue.pop();

	return frontViewEvent;
}

void ViewEventQueue::pushViewEvent(std::shared_ptr<ViewEvent> viewEvent)
{
	std::unique_lock lock(m_ViewEventMutex);
	m_ViewEventQueue.push(viewEvent);
}
