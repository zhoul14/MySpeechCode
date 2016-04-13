#ifndef WJQ_STATE_TRACE_FACTORY_H
#define WJQ_STATE_TRACE_FACTORY_H

#include "StateTrace.h"

class StateTraceFactory {
public:
	virtual StateTrace* getInstance() = 0;

	virtual void destroyInstance(StateTrace* st) = 0;
};


#endif