#ifndef WJQ_STATE_TRACE_H
#define WJQ_STATE_TRACE_H



class StateTrace {
	friend class SegmentAlgorithm;
	friend class SimpleSTFactory;
	friend class StateTraceBin;
protected:
	int enterTime;

	int cbidx;

	double lh;


	int refcnt;

	StateTrace* prev;

	int wordId;

	void set(int enterTime, int cbidx, double lh, int refcnt, StateTrace* prev, int wordId);

public:
	int getEnterTime() const;

	int getCodebookIndex() const;

	inline double getLikelihood() const {
		return this->lh;
	}

	StateTrace* getPrevious() const;

	int getWordId() const;

};

#endif