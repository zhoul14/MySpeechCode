#ifndef WJQ_WORD_NODE_H
#define WJQ_WORD_NODE_H
#include "../CommonLib/commonvars.h"

class WordNode {
	friend class NBestRecAlgorithm;
	friend class LatticeSearch;
	friend class TimeGridGen;
	friend class WordNodeFactory;
	friend class RecToken;
	friend bool wordNodeLargerThan(WordNode* i, WordNode* j);
private:

	//static const int MAX_BESTN = 20;

	int wordId;

	int endTime;

	int id;

	double endLh;

	int lhRank;

	WordNode* prev[BEST_N];

	double prevLhDiff[BEST_N];

	int jumpTime[7];

	int cbTrace[7];

	double lhTrace[7];

	int candPtr;

	long refCnt;

	long keyRefCnt;

	bool candTrimmed;

	void addPrev(WordNode* word);

	void addPrev(WordNode* word, double lhd);

	WordNode(const WordNode& other);

	WordNode();



	void reset();

	double wordLh();

	int prevNum();
};

bool wordNodeLargerThan(WordNode* i, WordNode* j);

#endif