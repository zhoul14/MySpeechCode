#ifndef WJQ_SEGMENT_ALGORITHM_H
#define WJQ_SEGMENT_ALGORITHM_H

#include <vector>
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../CommonLib/Dict/WordDict.h"
#include "StateTraceBin.h"
#include "StateTraceFactory.h"

struct SegmentResult {
	std::vector<int> frameLabel;

	std::vector<double> stateLh;

	double lh;

	SegmentResult() {
		lh = 0;
	}
};

struct SegmentUtility {
	WordDict* dict;
	GMMProbBatchCalc* bc;
	StateTraceFactory* factory;
};

class SegmentAlgorithm {
private:
	BinList binSet;

	int fDim;
	
	int ansNum;

	int* ansList;

	GMMProbBatchCalc* bc;

	WordDict* dict;

	StateTraceFactory* factory;

	int time;

	void constructBinSet();

	void constructDiBinSet();

	StateTrace* bestPreviousTrace(const StateTraceBin* bin) const;

	void frameSegmentation();

	void resetBinSet();

	void prune(StateTraceBin* bin);

public:
	SegmentResult segmentSpeech(int fnum, int fDim, int ansNum, const int* ansList, SegmentUtility util);

	SegmentAlgorithm();

	inline void setFactory(void* ft){
		factory = (StateTraceFactory*)ft;
	}

	~SegmentAlgorithm();

};


#endif