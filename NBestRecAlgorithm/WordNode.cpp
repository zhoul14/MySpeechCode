#include "WordNode.h"
#include <stdio.h>
#include <cstdlib>
#include <windows.h>

void WordNode::addPrev(WordNode* word) {
	addPrev(word, 0);
}

void WordNode::addPrev(WordNode* word, double lhd) {
	if (candPtr >= BEST_N) {
		printf("ptr[%d] out of range in ReckToken::pushBack, bestn = %d\n", candPtr, BEST_N);
		exit(-1);
	}

	prev[candPtr] = word;
	prevLhDiff[candPtr] = lhd;

	if (candPtr == 0)
		InterlockedIncrement(&word->keyRefCnt);
	InterlockedIncrement(&word->refCnt);
// 	if (candPtr == 0)
// 		word->keyRefCnt++;
// 	word->refCnt++;
	candPtr++;
		
}


WordNode::WordNode() {
	reset();
}

void WordNode::reset() {
	candPtr = 0;
	keyRefCnt = 0;
	refCnt = 0;
	for (int i = 0; i < BEST_N; i++) {
		prev[i] = NULL;
		prevLhDiff[i] = 0;
	}

}


double WordNode::wordLh() {
	WordNode* bestPrev = prev[0];
	if (bestPrev == NULL) {
		return this->endLh;
	}

	double r = this->endLh - bestPrev->endLh;
	return r;
}

int WordNode::prevNum() {
	return candPtr;
}

WordNode::WordNode(const WordNode& other) {
	wordId = other.wordId;
	endTime = other.endTime;
	endLh = other.endLh;
	candPtr = other.candPtr;
	refCnt = other.refCnt;
	keyRefCnt = other.keyRefCnt;


	for (int i = 0; i < candPtr; i++) {
		prevLhDiff[i] = other.prevLhDiff[i];
		prev[i] = other.prev[i];
	}

}

bool wordNodeLargerThan(WordNode* i, WordNode* j) { 
	return i->endLh > j->endLh;
}