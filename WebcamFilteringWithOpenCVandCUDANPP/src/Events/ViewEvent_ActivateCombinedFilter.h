#pragma once

#include "ViewEvent.h"

class ViewEvent_ActivateCombinedFilter:
	public ViewEvent
{
public:
	ViewEvent_ActivateCombinedFilter();

	void setActivateCombinedFilter(const bool activateCombinedFilter);
	bool getActivateCombinedFilter();

private:
	bool m_activateCombinedFilter;
};
