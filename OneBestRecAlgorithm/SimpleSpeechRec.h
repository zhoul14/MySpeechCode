#ifndef _WJQ_SIMPLE_SPEECH_REC_H_
#define _WJQ_SIMPLE_SPEECH_REC_H_

#include "STokenBin.h"
#include <vector>
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "SRecTokenFactory.h"
#include "../CommonLib/CommonVars.h"


class SimpleSpeechRec {
private:
	
	const WordDict* dict;

	std::vector<STokenBin*> binSet;

	SRecTokenFactory factory;

	int fDim;

	GMMProbBatchCalc* bc;

	int time;

	int cbNum;

	bool m_bHeadNOise;

	bool useSegmentModel;

	SRecToken* startNoiseNode;

	void constructBinSet();

	void releaseBinSet();

	void resetBinSet();

	void frameSegmentation();

	void timeZeroInit();

	void prune(STokenBin* bin);

	//WordNode* frameSynAlgorithm(int fnum, int fDim, const WordDict* dict, GMMProbBatchCalc* bc, int* allowJump);
	 bool isCrossWordCb(int cbType) {
		if (this->dict->triPhone)
		{
			return cbType == INITIAL0 || cbType == INITIAL0_C || cbType == TAIL_NOISE;
		}
		else 
			return cbType == DI_INITIAL0 || cbType == DI_TAIL_NOISE;
		
	}

	void lastJump();

public:

	//std::vector<std::vector<SWord> > recSpeech(int fnum, int fDim, const WordDict* dict, GMMProbBatchCalc* bc, int multiJump);

	SimpleSpeechRec(){
	}

	std::vector<SWord> recSpeech(int fnum, int fDim, WordDict* dict, GMMProbBatchCalc* bc, bool useSegmentModel,bool headnoise);


};

#endif