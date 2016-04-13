#ifndef _WJQ_NBEST_REC_ALGORITHM_H_
#define _WJQ_NBEST_REC_ALGORITHM_H_

#include "RecToken.h"
#include "TokenBin.h"
#include <vector>
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "WordNodeFactory.h"
#include "LatticeNode.h"

#include "../CommonLib/CommonVars.h"




class NBestRecAlgorithm {
private:
	static const int pruneThreshold = BEST_N;
	int candMergeStep(TokenBin* bin, RecToken** candToken,
		int candTokenNum, int maxCandNum, WordNode** resWord, double* resLhDiff);

	int tokenFilterStep(TokenBin* bin, int maxCandNum, RecToken** tokenCand);

	int createNewToken(int cbidx, int cbType, RecToken** candToken, int candTokenNum,
		WordNode** newCandWords, double* lhDiff, int preWordNum,
		RecToken* (*tokenBuffer)[BEST_N], int* newTokenCnt);

	const WordDict* dict;

	WordNodeFactory fact;

	std::vector<TokenBin*> binSet;

	bool useSegmentModel;

	bool fixedJumpTime;

	int fDim;

	GMMProbBatchCalc* bc;

	int time;

	int cbNum;

	int tokenCnt;

	int VJumpLimit;

	WordNode* startNoiseNode;

	void constructBinSet();

	void releaseBinSet();

	void resetBinSet();

	static const int FREE_JUMP = 0;
	static const int LIMITED_JUMP_ALLOW = 1;
	static const int LIMITED_JUMP_DISALLOW = 2;

	void frameSegmentation(int allowCrossWordJump = FREE_JUMP);

	void prune(TokenBin* bin);

	void timeZeroInit();

	WordNode* recSpeech(int fnum, int fDim, const WordDict* dict, GMMProbBatchCalc* bc, int* allowJump, bool useSegmentModel);

	std::vector<std::vector<SWord> > genAlignedResult(WordNode* l);

	Lattice rearrangeLattice(int fnum, WordNode* lastNode);

	void getJumpFlag(WordNode* x, int* flag, int fnum);

	void NBestRecAlgorithm::clearBinSet();

	bool isCrossWordCb(int cbType) {

		if (dict->triPhone)
		{
			return cbType == INITIAL0 || cbType == INITIAL0_C || cbType == TAIL_NOISE;
		}
		else
		{
			return cbType == INITIAL0 || cbType == DI_TAIL_NOISE;
		}

	}

	bool isInVowelDi(int cbType) {
		return isInVowel(cbType);
	}


	bool isInVowel(int cbType) {

		if (dict->triPhone)
		{
			return cbType != INITIAL0 && cbType != INITIAL0_C && cbType != TAIL_NOISE && cbType != INITIAL1;
		}
		else 
		{
			return cbType != DI_INITIAL0 && cbType != DI_TAIL_NOISE && cbType != DI_INITIAL1;
		}

	}
	static bool isInInitialDi(int cbType){
		return cbType == DI_INITIAL0 || cbType == DI_INITIAL1 || cbType == DI_TAIL_NOISE;
	}
	bool isInInitial(int cbType) {
		if(dict->triPhone)
			return cbType == INITIAL0 || cbType == INITIAL0_C || cbType == TAIL_NOISE || cbType == INITIAL1;
		else 
			return cbType == DI_INITIAL0 || cbType == DI_INITIAL1 || cbType == DI_TAIL_NOISE;
	}

	static int funcCallCnt;



public:



	std::vector<std::vector<SWord> > recSpeech(int fnum, int fDim, const WordDict* dict, GMMProbBatchCalc* bc, int multiJump, bool useSegmentModel);

	NBestRecAlgorithm(){
	}

};

#endif