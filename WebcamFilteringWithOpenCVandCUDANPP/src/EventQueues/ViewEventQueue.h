#pragma once

#include <memory>
#include <mutex>
#include <queue>

#include "Events/ViewEvent.h"


class ViewEventQueue
{
public:
	std::shared_ptr<ViewEvent> popViewEvent();
	void pushViewEvent(std::shared_ptr<ViewEvent> viewEvent);

private:
	std::mutex m_ViewEventMutex;

	std::queue<std::shared_ptr<ViewEvent>> m_ViewEventQueue;
};

