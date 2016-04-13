#ifndef WJQ_WORD_NODE_FACTORY_H
#define WJQ_WORD_NODE_FACTORY_H


//#include "boost/pool/pool.hpp"
#include "WordNode.h"
#include <windows.h>

class WordNodeFactory {
private:
	long createCnt;

	long destroyCnt;

	//boost::pool<> bp;

public:
	WordNode* getInstance() {
		WordNode* newNode;
		InterlockedIncrement(&createCnt);
		newNode = (WordNode*)malloc(sizeof(WordNode));
		newNode->reset();
		newNode->id = createCnt;
		return newNode;
	}

	void checkClear() {
		if (createCnt != destroyCnt) {
			printf("create_cnt=%d, destroy_cnt=%d\n", createCnt, destroyCnt);
			exit(-1);
		}
	}

	void unkeyPrune(WordNode* node) {
		if (node->keyRefCnt == 0 && !node->candTrimmed && node->prevNum() > 0) {
			for (int i = 1; i < node->prevNum(); i++) {
				destroyInstance(node->prev[i]);
				node->prev[i] = NULL;
			}
			node->candPtr = 1;
			node->candTrimmed = true;

			//node->prev[0]->keyRefCnt--;
			InterlockedDecrement(&node->prev[0]->keyRefCnt);
			unkeyPrune(node->prev[0]);

		}
	}

	void destroyInstance(WordNode* node) {
		//node->refCnt--;
		InterlockedDecrement(&node->refCnt);
		if (node->refCnt <= 0) {
			for (int i = 0; i < node->prevNum(); i++) {
				WordNode* prev = node->prev[i];
				destroyInstance(prev);
			}

			//bp.free(node);
			//delete node;
			free(node);
			//destroyCnt++;
			InterlockedIncrement(&destroyCnt);
			return;
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

	WordNodeFactory() {
		destroyCnt = 0;
		createCnt = 0;
	}

};


#endif