#ifndef WJQ_S_REC_TOKEN_FACTORY_H
#define WJQ_S_REC_TOKEN_FACTORY_H

//#include "boost/pool/pool.hpp"
#include "STokenBin.h"
#include <windows.h>


class SRecTokenFactory {
private:
	long createCnt;

	long destroyCnt;

	long id;

//	boost::pool<> bp;

public:
	SRecToken* getInstance() {
		InterlockedIncrement(&createCnt);
		//SRecToken* newSt = (SRecToken*)bp.malloc();
		SRecToken* newSt = new SRecToken();
		newSt->init(id);
		
		InterlockedIncrement(&id);
		return newSt;
	}

	void destroyInstance(SRecToken* st) {
		InterlockedDecrement(&st->refcnt);
		if (st->refcnt <= 0) {
			SRecToken* prev = st->prev;

			//bp.free(st);
			delete st;
			InterlockedIncrement(&destroyCnt);

			if (prev != NULL) {
				destroyInstance(prev);
			}
		}
	}

	int getAllocatedNum() {
		return createCnt - destroyCnt;
	}

	int getCreateCnt() {
		return createCnt;
	}

	int getDestroyCnt() {
		return destroyCnt;
	}

//	SRecTokenFactory() : bp(sizeof(SRecToken)) {
	SRecTokenFactory() {
		destroyCnt = 0;
		createCnt = 0;
		id = 0;
	}

};


#endif