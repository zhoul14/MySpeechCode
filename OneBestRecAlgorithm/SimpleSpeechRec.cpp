#include "SimpleSpeechRec.h"
#include <omp.h>

void SimpleSpeechRec::constructBinSet() {
	if (binSet.size() != 0) {
		printf("binSet.size() != 0, the recognizer is not in a clear state\n");
		exit(-1);
	}

	for (int i = 0; i < cbNum; i++) {
		STokenBin* bin = new STokenBin(i);
		bin->isEndingBin = false;
		binSet.push_back(bin);
	}

	for (int i = 0; i < cbNum; i++) {
		std::vector<int> jmpList = dict->getJumpList(i);
		for (int j = 0; j < jmpList.size(); j++) {
			STokenBin* prev = binSet.at(jmpList.at(j));
			binSet.at(i)->addPreviousBin(prev);
		}
	}

	STokenBin* endingBin = new STokenBin(-1);
	endingBin->isEndingBin = true;
	if(dict->triPhone)
		for (int i = 0; i < V_CLASS_NUM; i++) {
			for (int j = 0; j < C_CLASS_NUM; j++) {
				int cbidx = dict->getCACbId(i, j, FINAL3_C);
				endingBin->addPreviousBin(binSet.at(cbidx));
			}
		}

		for (int i = 0; i < FINAL_WORD_NUM; i++) {
			int cbidx = dict->getIsoCbId(i, FINAL3);
			if (!dict->triPhone)
			{
				cbidx = dict->getDiIsoCbId(i,DI_FINAL3);
			}
			endingBin->addPreviousBin(binSet.at(cbidx));
		}

		endingBin->addPreviousBin(binSet.at(dict->getNoiseCbId()));

		binSet.push_back(endingBin);
}

void SimpleSpeechRec::releaseBinSet() {
	int cnt = 0;
	for (auto i = binSet.begin(); i != binSet.end(); i++) {
		std::vector<SRecToken*>* stlist = &((*i)->content);
		for (auto j = stlist->begin(); j != stlist->end(); j++) {
			//(*j)->releaseCand(&fact);
			factory.destroyInstance(*j);
			//delete *j;
		}
		delete *i;
		cnt++;
	}
	binSet.clear();
}

void SimpleSpeechRec::resetBinSet() {
	for (int i = 0; i < binSet.size(); i++) {
		std::vector<SRecToken*>* stlist = &(binSet.at(i)->content);
		for (int j = 0; j < stlist->size(); j++) {
			SRecToken* t = stlist->at(j);
			factory.destroyInstance(t);
		}
		stlist->clear();
	}
}

void SimpleSpeechRec::frameSegmentation() {
	time++;
	if (time == 0) {
		timeZeroInit();
		return;
	}

	SRecToken** newTokenBuffer = new SRecToken*[cbNum];
	for (int i = 0; i < cbNum; i++) {
		newTokenBuffer[i] = NULL;
	}

	int* cbTypeLookup = new int[cbNum];
	for (int i = 0; i < cbNum; i++) {
		cbTypeLookup[i] = dict->getCbType(i);
	}

	omp_set_dynamic(true);
#pragma omp parallel for 
	for (int i = 0; i < cbNum; i++) {
		STokenBin* bin = binSet[i];
		SRecToken* candToken = bin->getPreviousBest();
		if (!candToken)
			continue;
		int cbType = (i == cbNum-1) ? DI_TAIL_NOISE: cbTypeLookup[i];
		bool isCrossWord = isCrossWordCb(cbType);



		SRecToken* candWord = NULL;
		SRecToken* newToken = factory.getInstance();
		//newToken->copyFrom(candToken);
		if (isCrossWord) {
			candWord = factory.getInstance();
			candWord->copyFrom(candToken);
			if (candToken->prev)
				InterlockedIncrement(&candToken->prev->refcnt);
			candWord->endTime = time;
		} else {
			candWord = candToken->prev;
			newToken->CId = candToken->CId;
			newToken->VId = candToken->VId;
			newToken->wordId = candToken->wordId;
			if (dict->triPhone) {	
				if (cbType == INITIAL1) {   
					newToken->CId = dict->getCVIdFromCbId(i);
				} else if (cbType == FINAL0) {
					newToken->VId = dict->getCVIdFromCbId(i);
					newToken->wordId = dict->getWordIdFromCVLink(candToken->currentCbId, i);
				}
			}
			else{
				if (cbType == DI_INITIAL1) {
					newToken->CId = dict->getCVIdFromCbId(i);
				} else if (cbType == DI_FINAL0) {
					newToken->VId = dict->getCVIdFromCbId(i);
					newToken->wordId = dict->getWordIdFromCVLink(candToken->currentCbId, i);
				}
			}


		}
		//
		newToken->currentCbId = i;
		newToken->dur = 1;



		double durLh = useSegmentModel ? bc->getDurLh(i, 1) : 0;
		double stateLh = bc->getStateLh(i, time);
		newToken->lh = candToken->lh + durLh + stateLh;
		newToken->prev = candWord;
		InterlockedIncrement(&candWord->refcnt);

		newTokenBuffer[i] = newToken;
	}

	//状态驻留
	for (int i = 0; i < cbNum; i++) {
		STokenBin* bin = binSet.at(i);

		int k =i;
		if(m_bHeadNOise&&i == cbNum-1)
			k=  dict->noiseId;

		double stateLh = bc->getStateLh(k, time);

		for (auto j = bin->content.begin(); j != bin->content.end(); j++) {
			SRecToken* t = *j;

			t->dur += 1;
			int dur = t->dur;
			double deltaDurLh = useSegmentModel ? bc->getDurLhDelta(k, dur) : 0;

			t->lh += deltaDurLh + stateLh;
		}
	}

	//完成状态跳转
	for (int i = 0; i < cbNum; i++) {
		STokenBin* bin = binSet.at(i);
		if (newTokenBuffer[i] != NULL)
			bin->addToken(newTokenBuffer[i]);
		prune(bin);
	}

	delete [] cbTypeLookup;
	delete [] newTokenBuffer;
}

void SimpleSpeechRec::timeZeroInit() {

	//在全部非协同发音I0状态中放入token
	for (int i = 0; i < INITIAL_WORD_NUM; i++) {
		int cbidx = dict->getIsoCbId(i, INITIAL0);
		double durLh = useSegmentModel ? bc->getDurLh(cbidx, 1) : 0;
		double stateLh = bc->getStateLh(cbidx, time);

		SRecToken* token = factory.getInstance();
		token->lh = durLh + stateLh;
		token->currentCbId = cbidx;
		token->dur = 1;
		token->prev = startNoiseNode;
		startNoiseNode->refcnt++;
		//token->addPrev(startNoiseNode, 0);

		binSet.at(cbidx)->addToken(token);
	}

	//在全部协同发音I0状态中放入token
	if(dict->triPhone)
		for (int i = 0; i < C_CLASS_NUM; i++) {
			for (int j = 0; j < V_CLASS_NUM; j++) {
				int cbidx = dict->getCACbId(i, j, INITIAL0_C);
				double durLh = useSegmentModel ? bc->getDurLh(cbidx, 1) : 0;
				double stateLh = bc->getStateLh(cbidx, time);

				SRecToken* token = factory.getInstance();
				token->lh = durLh + stateLh;
				token->currentCbId = cbidx;
				token->dur = 1;
				token->prev = startNoiseNode;
				startNoiseNode->refcnt++;
				//token->addPrev(startNoiseNode, 0);

				binSet.at(cbidx)->addToken(token);
			}
		}

		//在噪声状态中放入token
		int noiseIdx = dict->getNoiseCbId();
		double durLh = useSegmentModel ? bc->getDurLh(noiseIdx, 1) : 0;
		double stateLh = bc->getStateLh(noiseIdx, time);

		SRecToken* token = factory.getInstance();
		token->lh = durLh + stateLh;
		token->currentCbId = noiseIdx;
		token->dur = 1;

		binSet.at(noiseIdx)->addToken(token);
		//加入头噪声

		if (m_bHeadNOise)
		{
			SRecToken* token2 = factory.getInstance();
			token2->lh = token->lh;
			token2->currentCbId = noiseIdx+1;
			token2->dur = 1;
			token2->prev = startNoiseNode;
			startNoiseNode->refcnt++;
			binSet.at(noiseIdx+1)->addToken(token2);
		}
}

void SimpleSpeechRec::prune(STokenBin* bin) {
	auto& list = bin->content;
	if (list.size() < 2)
		return;

	if (useSegmentModel) {
		int i = list.size() - 1;
		int iprev = i - 1;
		while (iprev >= 0) {
			if (list[iprev]->lh < list[i]->lh) {
				factory.destroyInstance(list[iprev]);
				list.erase(list.begin() + iprev);
			}
			i--;
			iprev--;

		}
	} else {
		int bestIdx = 0;
		double bestLh = list[0]->lh;
		for (int i = 1; i < list.size(); i++) {
			if (list[i]->lh > bestLh) {
				bestIdx = i;
				bestLh = list[i]->lh;
			}
		}
		for (int i = 0; i < list.size(); i++) {
			if (i == bestIdx)
				continue;
			factory.destroyInstance(list[i]);
		}
		SRecToken* bestToken = list[bestIdx];
		list.clear();
		list.push_back(bestToken);
	}

}

void SimpleSpeechRec::lastJump() {
	STokenBin* bin = binSet[cbNum];
	SRecToken* candToken = bin->getPreviousBest();
	if (!candToken) {
		printf("error, no token reach the last bin\n");
		exit(-1);
	}

	SRecToken* candWord = factory.getInstance();;
	SRecToken* newToken = factory.getInstance();
	newToken->prev = candToken;

	candWord->copyFrom(candToken);
	if (candToken->prev)
		candToken->prev->refcnt++;
	candWord->endTime = time;

	newToken->prev = candWord;
	candWord->refcnt++;

	bin->addToken(newToken);
}

std::vector<SWord> SimpleSpeechRec::recSpeech(int fnum, int fDim, WordDict* dict, GMMProbBatchCalc* bc, bool useSegmentModel,bool bHeadnoise) {
	this->fDim = fDim;
	this->dict = dict;
	this->bc = bc;
	this->cbNum = dict->getTotalCbNum();
	this->useSegmentModel = useSegmentModel;
	this->m_bHeadNOise = bHeadnoise;
	if (bHeadnoise)
	{
		cbNum++;
	}
	startNoiseNode = factory.getInstance();
	constructBinSet();
	time = -1;
	for (int i = 0; i < fnum; i++) {
		frameSegmentation();
	}
	lastJump();

	std::vector<SWord> res0;
	SRecToken* t = binSet[cbNum]->getBestToken();
	SRecToken* p = t->prev;
	while (p) {
		if (p->wordId >= 0) {
			SWord s;
			s.endTime = p->endTime;
			s.lh = p->lh;
			s.wordId = p->wordId;
			res0.push_back(s);
		}
		p=p->prev;
	}

	releaseBinSet();
	int allNum = factory.getAllocatedNum();
	if (allNum != 0) {
		printf("SRecToken Number[%d] != 0\n", allNum);
		exit(-1);
	}

	std::vector<SWord> res;
	for (auto i = res0.rbegin(); i != res0.rend(); i++) {
		res.push_back(*i);
	}
	for (auto i = res.begin(); i != res.end(); i++) {
		i->label = dict->wordToText(i->wordId);
	}
	return res;
}