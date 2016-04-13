#ifndef WJQ_S_TOKEN_BIN_H
#define WJQ_S_TOKEN_BIN_H

#include <vector>
#include <iostream>


#include "../CommonLib/commonvars.h"
#include "../CommonLib/Dict/WordDict.h"

struct SRecToken {
	double lh;
	int CId;
	int VId;
	int wordId;
	int endTime;
	int currentCbId;
	int dur;
	long refcnt;
	long id;
	SRecToken* prev;

	void init(long id) {
		CId = -1;
		VId = -1;
		wordId = -1;
		endTime = -1;
		dur = -1;
		currentCbId = -1;
		prev = NULL;
		refcnt = 0;
		lh = 0;
		this->id = id;
	}

	void copyFrom(SRecToken* t) {
		CId = t->CId;
		VId = t->VId;
		wordId = t->wordId;
		endTime = t->endTime;
		dur = t->dur;
		currentCbId = t->currentCbId;
		prev = t->prev;
		refcnt = t->refcnt;
		lh = t->lh;
	}
	
	void print() {
		printf("id = %d, CId = %d, VId = %d, wordId = %d, endTime = %d, dur = %d, refcnt = %d, prev = %x\n", id, CId, VId, wordId, endTime, dur, refcnt, prev);
	}
};

struct STokenBin {
	

	int cbidx;

	bool isEndingBin;

	std::vector<SRecToken*> content;

	std::vector<STokenBin*> previous;

	void addToken(SRecToken* t) {
		if (t == NULL) {
			std::cout << "state trace pointer is null";
			exit(-1);
		}
		content.push_back(t);
	}

	STokenBin(int cbidx) {
		this->cbidx = cbidx;
		isEndingBin = false;
	}

	void addPreviousBin(STokenBin* b) {
		if (b == NULL) {
			printf("state trace bin pointer is null\n");
			exit(-1);
		}
		previous.push_back(b);
	}

	SRecToken* getBestToken() {
		return content.size() == 0 ? NULL : content.front();
	}

	SRecToken* getPreviousBest() {
		if (previous.size() == 0) {
			return NULL;
		}

		SRecToken* best = NULL;
		for (auto i = previous.begin(); i != previous.end(); i++) {
			SRecToken* t = (*i)->getBestToken();
			if (t == NULL)
				continue;

			if (best == NULL || t->lh > best->lh) {
				best = t;
			}
		}
		return best;
	}
};


#endif