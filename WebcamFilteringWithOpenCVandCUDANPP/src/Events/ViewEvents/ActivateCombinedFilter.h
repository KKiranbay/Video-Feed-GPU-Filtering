#pragma once

#include "Events/ViewEvents/ViewEvent.h"


class ActivateCombinedFilter:
	public ViewEvent
{
public:
	ActivateCombinedFilter();

	void setActivateCombinedFilter(const bool activateCombinedFilter);
	bool getActivateCombinedFilter();

private:
	bool m_activateCombinedFilter;
};
