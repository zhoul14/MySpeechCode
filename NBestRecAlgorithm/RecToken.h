#ifndef _WJQ_REC_TOKEN_H_
#define _WJQ_REC_TOKEN_H_

#include <vector>
#include "WordNode.h"
#include "WordNodeFactory.h"



class RecToken {
	friend class NBestRecAlgorithm;
private:

	static bool tokenCmp(RecToken* a, RecToken* b) {
		return (a->lh > b->lh);
	}

	static const int UNINITIALIZED_WID = -10;

	static const int UNINITIALIZED_CID = -10;

	static const int UNINITIALIZED_VID = -10;

	static const int NOISE_WID = -1;

	static int count;

	int id;

	WordNode* candWord[BEST_N];

	double lhDiff[BEST_N];

	int ptr;

	double lh;

	int wordId;

	int CId;

	int VId;

	int currentCbId;

	int dur;

// 	int noiseDur;
// 
// 	double noiseLh;

	int jumpTime[7];

	int cbTrace[7];
	
	double lhTrace[7];
	//std::vector<int> historyCbId;	//only for debug;

	int candWordNum();

	RecToken();

	void releaseCand(WordNodeFactory* fact);

	void addPrev(WordNode* word, double lhd);

	void setJumpTimeAndCb(int cbType, int cbNum, int t, double lh, bool bTriPhone);

};



#endif