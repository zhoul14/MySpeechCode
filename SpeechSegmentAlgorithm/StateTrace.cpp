#include "StateTrace.h"


int StateTrace::getEnterTime() const {
	return this->enterTime;
}

int StateTrace::getCodebookIndex() const {
	return this->cbidx;
}

StateTrace* StateTrace::getPrevious() const {
	return this->prev;
}

void StateTrace::set(int enterTime, int cbidx, double lh, int refcnt, StateTrace* prev, int wordId) {
	this->enterTime = enterTime;
	this->cbidx = cbidx;
	this->lh = lh;
	this->refcnt = refcnt;
	this->prev = prev;
	this->wordId = wordId;
}

int StateTrace::getWordId() const {
	return wordId;
}