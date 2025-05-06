#pragma once

enum class ViewEventTypesEnum
{
	ActivateCombinedFilter,
	ChangeActiveFilters,
	ChangeActiveFiltersOnCombinedFilter,
	None
};

class ViewEvent
{
public:
	ViewEvent();
	virtual ~ViewEvent();

	const ViewEventTypesEnum& getViewEventType();

protected:
	ViewEvent(const ViewEventTypesEnum& eventType);

	ViewEventTypesEnum m_ViewEventType;
};
