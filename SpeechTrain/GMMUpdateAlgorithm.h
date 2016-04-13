#ifndef WJQ_GMM_UPDATE_ALGORITHM_H
#define WJQ_GMM_UPDATE_ALGORITHM_H

#include <vector>
#include "GMMEstimator.h"
#include "../CommonLib/SpeechFrame.h"

typedef std::vector<SpeechFrame> FrameVec;

struct GMMUpdateInfo {
	
	bool isSuccess;

	int frameNumForTrain;

	std::vector<GMMIterInfo> iterInfo;
};

class GMMUpdateAlogrithm {
private:

	GMMCodebookSet* codebooks;

	GMMEstimator* estimator;

	FrameVec* classifiedFrames;

	double* firstMoment;

	double* secondMoment;

	int* durCnt;

	bool frameCntSufficient(int n);

public:

	GMMUpdateAlogrithm(GMMCodebookSet* codebooks, int maxIter);

	void collect(std::vector<int> frameLabel, double* frames);

	std::vector<GMMUpdateInfo> update();

	~GMMUpdateAlogrithm();
};

#endif