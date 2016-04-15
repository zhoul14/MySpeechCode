#include "RecToken.h"
#include "../CommonLib/commonvars.h"
#include <windows.h>
int RecToken::count = 0;

int RecToken::candWordNum() {
	return ptr;
}
void RecToken::setJumpTimeAndCb(int cbType, int cbNum, int t, double lh, bool bTriPhone) {
	int idx = 0;
	if (bTriPhone)
	{
		if (cbType == INITIAL0 || cbType == INITIAL0_C) {
			idx = 0;
		} else if (cbType == INITIAL1) {
			idx = 1;
		} else if (cbType == FINAL0) {
			idx = 2;
		} else if (cbType == FINAL1) {
			idx = 3;
		} else if (cbType == FINAL2 || cbType == FINAL2_C) {
			idx = 4;
		} else if (cbType == FINAL3 || cbType == FINAL3_C) {
			idx = 5;
		} else if (cbType == TAIL_NOISE) {
			idx = 6;
		} else {
			printf("unknown cbType\n");
			exit(-1);
		}
	}
	else
	{
		if (cbType == DI_INITIAL0) {
			idx = 0;
		} else if (cbType == DI_INITIAL1) {
			idx = 1;
		} else if (cbType == DI_FINAL0) {
			idx = 2;
		} else if (cbType == DI_FINAL1) {
			idx = 3;
		} else if (cbType == DI_FINAL2) {
			idx = 4;
		} else if (cbType == DI_FINAL3) {
			idx = 5;
		} else if (cbType == DI_TAIL_NOISE) {
			idx = 6;
		} else {
			printf("unknown cbType\n");
			exit(-1);
		}
	}
	jumpTime[idx] = t;
	cbTrace[idx] = cbNum;
	lhTrace[idx] = lh;
}

RecToken::RecToken() {
	ptr = 0;

	id = count;
	count++;
	wordId = UNINITIALIZED_WID;
	CId = UNINITIALIZED_CID;
	VId = UNINITIALIZED_VID;
	// 	for (int i = 0; i < 7; i++) {
	// 		jumpTime[i] = -1;
	// 	}

}

void RecToken::releaseCand(WordNodeFactory* fact) {
	if (ptr > 0) {
		candWord[0]->keyRefCnt--;
		fact->unkeyPrune(candWord[0]);
	}
	for (int i = 0; i < ptr; i++) {
		fact->destroyInstance(candWord[i]);
	}
}

void RecToken::addPrev(WordNode* word, double lhd) {
	if (ptr >= BEST_N) {
		printf("ptr[%d] out of range in ReckToken::addPrev, bestn = %d\n", ptr, BEST_N);
		exit(-1);
	}

	candWord[ptr] = word;
	lhDiff[ptr] = lhd;

	if (ptr == 0)
		InterlockedIncrement(&word->keyRefCnt);
	InterlockedIncrement(&word->refCnt);

	// 	if (ptr == 0)
	// 		word->keyRefCnt++;
	// 	word->refCnt++;

	ptr++;
}