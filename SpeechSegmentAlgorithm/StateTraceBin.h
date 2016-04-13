#ifndef WJQ_STATE_TRACE_BIN_H
#define WJQ_STATE_TRACE_BIN_H

#include <vector>
#include <iostream>

#include "StateTrace.h"
#include "StateTraceFactory.h"
#include "../CommonLib/commonvars.h"
#include "../CommonLib/Dict/WordDict.h"

class StateTraceBin;
class SegmentAlgorithm;

typedef std::vector<StateTrace*> StateTraceList;
typedef std::vector<StateTraceBin*> BinList;


class StateTraceBin {
	friend class SegmentAlgorithm;

	friend class NBestRecAlgorithm;

private:
	int cbidx;
	
	StateTraceList content;

	BinList previous;

	void addStateTrace(StateTrace* s);

	void addPreviousBin(StateTraceBin* b);

	StateTraceBin(int cbidx);

	StateTrace* getBestTrace();

	//void prune(StateTraceFactory* factory);

	std::vector<StateTrace*> getPreviousBest(int maxStNum, bool distinctSylCheck, WordDict* dict);

	int stateTraceNum();

};


#endif