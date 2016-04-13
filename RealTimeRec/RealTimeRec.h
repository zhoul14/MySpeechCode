#ifndef WJQ_REAL_TIME_REC_H
#define WJQ_REAL_TIME_REC_H
#include "../OneBestRecAlgorithm/SimpleSpeechRec.h"
#include "../NBestRecAlgorithm/NBestRecAlgorithm.h"
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../CommonLib/Dict/WordDict.h"
#include <vector>

class RealTimeRec {
private:
	SimpleSpeechRec* oneBestRec;

	NBestRecAlgorithm* nBestRec;

	GMMCodebookSet* set;

	WordDict* dict;

	GMMProbBatchCalc* gbc;

	bool useCuda;

	bool useSegmentModel;

	bool useNBest;
public:
	RealTimeRec(const char* codebookFile, const char* dictFile, double durWeight, bool useCuda, bool useSegmentModel, bool useNBest);

	std::vector<std::vector<SWord> > recSpeech(short* samples, int sampleNum);

	~RealTimeRec();
};


#endif