#ifndef WJQ_SIMPLE_ST_FACTORY_H
#define WJQ_SIMPLE_ST_FACTORY_H

#include "StateTraceFactory.h"
//#include "boost/pool/pool.hpp"

class SimpleSTFactory : public StateTraceFactory {
private:
	int createCnt;

	int destroyCnt;

//	boost::pool<> bp;

public:
	virtual StateTrace* getInstance() {
		createCnt++;
		StateTrace* newSt = (StateTrace*)malloc(sizeof(StateTrace));
		//StateTrace* newSt = (StateTrace*)bp.malloc();
		return newSt;
	}

	virtual void destroyInstance(StateTrace* st) {
		if (st->refcnt == 0) {
			StateTrace* prev = st->getPrevious();
			free(st);
			//bp.free(st);
			destroyCnt++;

			if (prev != NULL) {
				prev->refcnt--;
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

	//SimpleSTFactory() : bp(sizeof(StateTrace)) {
	SimpleSTFactory() {
		destroyCnt = 0;
		createCnt = 0;
	}

};


#endif