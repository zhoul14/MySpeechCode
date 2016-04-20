#include "SegmentAlgorithm.h"
#include "../CommonLib/commonvars.h"

void SegmentAlgorithm::resetBinSet() {

	for (auto i = binSet.begin(); i != binSet.end(); i++) {
		StateTraceList* stlist = &((*i)->content);
		for (auto j = stlist->begin(); j != stlist->end(); j++) {
			factory->destroyInstance(*j);
		}
		delete *i;
	}
	binSet.clear();
}

SegmentAlgorithm::SegmentAlgorithm() {
	ansList = NULL;
	dict = NULL;
	bc = NULL;
	factory = NULL;
}


//I0-->I1-->F0-->F1-->F2  -->  F3 -->SIL(optional)-->I0
//     ^              |
//I0C--|              `->F2C-->F3C         -->       I0C

void SegmentAlgorithm::constructBinSet() {
	if (binSet.size() != 0) {
		std::cout << "binSet.size() != 0, the trainner is not in a clear state";
		exit(-1);
	}

	int noiseId = dict->getNoiseCbId();
	StateTraceBin* firstSil = new StateTraceBin(noiseId);
	binSet.push_back(firstSil);	//句首静音

	StateTraceBin* lastSil = NULL;
	StateTraceBin* lastF3 = NULL;
	StateTraceBin* lastF3C = NULL;

	int cbInfo[STATE_NUM_IN_WORD];
	for (int i = 0; i < ansNum; i++) {
		int prevWord = i == 0 ? NO_PREVIOUS_WORD : ansList[i - 1];
		int nextWord = i == ansNum - 1 ? NO_NEXT_WORD : ansList[i + 1];
		int word = ansList[i];

		dict->getWordCbId(word, prevWord, nextWord, cbInfo);

		StateTraceBin* I0 = new StateTraceBin(cbInfo[INITIAL0]);
		if (i == 0) {
			I0->addPreviousBin(firstSil);
		} else {
			I0->addPreviousBin(lastSil);
			I0->addPreviousBin(lastF3);
		}
		binSet.push_back(I0);

		StateTraceBin* I0C = new StateTraceBin(cbInfo[INITIAL0_C]);
		if (i != 0) {
			I0C->addPreviousBin(lastF3C);
		}
		binSet.push_back(I0C);

		StateTraceBin* I1 = new StateTraceBin(cbInfo[INITIAL1]);
		I1->addPreviousBin(I0);
		I1->addPreviousBin(I0C);
		binSet.push_back(I1);

		StateTraceBin* F0 = new StateTraceBin(cbInfo[FINAL0]);
		F0->addPreviousBin(I1);
		binSet.push_back(F0);

		StateTraceBin* F1 = new StateTraceBin(cbInfo[FINAL1]);
		F1->addPreviousBin(F0);
		binSet.push_back(F1);

		StateTraceBin* F2 = new StateTraceBin(cbInfo[FINAL2]);
		F2->addPreviousBin(F1);
		binSet.push_back(F2);

		StateTraceBin* F2C = new StateTraceBin(cbInfo[FINAL2_C]);
		F2C->addPreviousBin(F1);
		binSet.push_back(F2C);

		StateTraceBin* F3 = new StateTraceBin(cbInfo[FINAL3]);
		F3->addPreviousBin(F2);
		binSet.push_back(F3);

		StateTraceBin* F3C = new StateTraceBin(cbInfo[FINAL3_C]);
		F3C->addPreviousBin(F2C);
		binSet.push_back(F3C);

		StateTraceBin* sil = new StateTraceBin(noiseId);
		sil->addPreviousBin(F3);
		binSet.push_back(sil);

		lastSil = sil;
		lastF3 = F3;
		lastF3C = F3C;
	}
}

void SegmentAlgorithm::constructDiBinSet() {
	if (binSet.size() != 0) {
		std::cout << "binSet.size() != 0, the trainner is not in a clear state";
		exit(-1);
	}

	int noiseId = dict->getNoiseCbId();
	StateTraceBin* firstSil = new StateTraceBin(noiseId);
	binSet.push_back(firstSil);	//句首静音

	StateTraceBin* lastSil = NULL;
	StateTraceBin* lastF3 = NULL;

	int cbInfo[STATE_NUM_IN_DI];
	for (int i = 0; i < ansNum; i++) {
		int prevWord = i == 0 ? NO_PREVIOUS_WORD : ansList[i - 1];
		int nextWord = i == ansNum - 1 ? NO_NEXT_WORD : ansList[i + 1];
		int word = ansList[i];

		dict->getDiWordCbId(word, cbInfo);

		StateTraceBin* I0 = new StateTraceBin(cbInfo[DI_INITIAL0]);
		if (i == 0) {
			I0->addPreviousBin(firstSil);
		} else {
			I0->addPreviousBin(lastSil);
			I0->addPreviousBin(lastF3);
		}
		binSet.push_back(I0);


		StateTraceBin* I1 = new StateTraceBin(cbInfo[DI_INITIAL1]);
		I1->addPreviousBin(I0);
		binSet.push_back(I1);

		StateTraceBin* F0 = new StateTraceBin(cbInfo[DI_FINAL0]);
		F0->addPreviousBin(I1);
		binSet.push_back(F0);

		StateTraceBin* F1 = new StateTraceBin(cbInfo[DI_FINAL1]);
		F1->addPreviousBin(F0);
		binSet.push_back(F1);

		StateTraceBin* F2 = new StateTraceBin(cbInfo[DI_FINAL2]);
		F2->addPreviousBin(F1);
		binSet.push_back(F2);

		StateTraceBin* F3 = new StateTraceBin(cbInfo[DI_FINAL3]);
		F3->addPreviousBin(F2);
		binSet.push_back(F3);

		StateTraceBin* sil = new StateTraceBin(noiseId);
		sil->addPreviousBin(F3);
		binSet.push_back(sil);

		lastSil = sil;
		lastF3 = F3;
	}
}


StateTrace* SegmentAlgorithm::bestPreviousTrace(const StateTraceBin* bin) const {
	
	if (bin->previous.size() == 0) {
		return NULL;
	}

	StateTrace* best = NULL;
	for (auto i = bin->previous.begin(); i != bin->previous.end(); i++) {
		StateTrace* t = (*i)->getBestTrace();
		if (t == NULL)
			continue;

		if (best == NULL) {
			best = t;
		} else {
			if (t->lh > best->lh) {
				best = t;
			}
		}
	}
	return best;
}

void SegmentAlgorithm::prune(StateTraceBin* bin) {
	StateTraceList& list = bin->content;
	if (list.size() < 2)
		return;

	if (bc->ifUseSegmentModel()) {
		int i = list.size() - 1;
		int iprev = i - 1;
		while (iprev >= 0) {
			if (list[iprev]->lh < list[i]->lh) {
				factory->destroyInstance(list[iprev]);
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
			factory->destroyInstance(list[i]);
		}
		StateTrace* bestToken = list[bestIdx];
		list.clear();
		list.push_back(bestToken);
	}

}

SegmentResult SegmentAlgorithm::segmentSpeech(int fnum, int fDim, int ansNum, const int* ansList, SegmentUtility util) {
	
	if (fnum < ansNum * HMM_STATE_DUR * 2) {
		SegmentResult res;
		return res;
	}
	
	this->bc = util.bc;
	this->ansNum = ansNum;
	if (this->ansList != NULL)
		delete [] this->ansList;
	this->ansList = new int[ansNum];

	for (int i = 0; i < ansNum; i++) {
		this->ansList[i] = ansList[i];
	}
	this->fDim = fDim;
	this->dict = util.dict;

	if (!factory)
	{
		this->factory = util.factory;
	}

	time = -1;
	if(dict->triPhone)
		constructBinSet();
	else
		constructDiBinSet();
	for (int i = 0; i < fnum; i++) {
		frameSegmentation();
	}

	int len = binSet.size();

	StateTrace* lastSil = binSet.at(len - 1)->getBestTrace();
	StateTrace* lastF3 = binSet.at(len - 3)->getBestTrace();
	
	if (lastSil == NULL) {
		std::cout << "the sil bin of last word is empty";
		exit(-1);
	}

	if (lastF3 == NULL) {
		std::cout << "the f3 bin of last word is empty";
		exit(-1);
	}

	SegmentResult res;
	StateTrace* best = lastSil->lh > lastF3->lh ? lastSil : lastF3;
	res.lh = best->lh;

	std::vector<int> t;
	std::vector<double> tlh;

	StateTrace* i = best;
	int rtime = fnum;
	while (i != NULL) {
		rtime--;
		t.push_back(i->cbidx);
		tlh.push_back(i->lh);
		if (rtime ==  i->enterTime) {
			i = i->prev;
		}
	}
	
	for (auto i = t.rbegin(); i != t.rend(); i++) {
		res.frameLabel.push_back(*i);
	}

	for (auto i = tlh.rbegin(); i != tlh.rend(); i++) {
		res.stateLh.push_back(*i);
	}

	resetBinSet();

	return res;
}

void SegmentAlgorithm::frameSegmentation() {
	time++;

	//0时刻放入初值
	if (time == 0) {
		StateTrace* s0 = factory->getInstance();

		int StateLen=dict->triPhone?STATE_NUM_IN_WORD:STATE_NUM_IN_DI;
			
		int *cb=new int[StateLen];
		if(dict->triPhone)
			dict->getWordCbId(ansList[0], NO_PREVIOUS_WORD, NO_NEXT_WORD, cb);
		else
			dict->getDiWordCbId(ansList[0],cb);

		//种子放入第一字的I0状态
		int cbidx0 = cb[INITIAL0];
		double stateLh0 = bc->getStateLh(cbidx0, time);
		double durLh0 = bc->getDurLh(cbidx0, 1);
		s0->set(time, cbidx0, stateLh0 + durLh0, 0, NULL, -1);
		binSet.at(1)->addStateTrace(s0);

		//种子放入句首的sil
		StateTrace* s1 = factory->getInstance();
		int cbidx1 = dict->getNoiseCbId();
		double stateLh1 = bc->getStateLh(cbidx1, time);
		double durLh1 =bc->getDurLh(cbidx1, 1);
		s1->set(time, cbidx1, stateLh1 + durLh1, 0, NULL, -1);
		binSet.at(0)->addStateTrace(s1);

		delete []cb;

		return;
	}

	//time != 0

	for (auto i = binSet.rbegin(); i != binSet.rend(); i++) {
		StateTraceBin* b = *i;
		int cbidx = b->cbidx;
		//在一句话的最后一字可能发生这样的情况，因为不存在下一字，无法协同发音，协同发音的状态ID的值为INVALID_STATE_ID
		if (cbidx == WordDict::INVALID_STATE_ID) 
			continue;

		double stateLh = bc->getStateLh(cbidx, time);


		//处理状态内跳转
		for (int j = 0; j < b->content.size(); j++) {
			StateTrace* st = b->content[j];
			int dur = time - st->enterTime + 1;

			double durLhDelta = bc->getDurLhDelta(cbidx, dur);
			double newLh = st->lh + durLhDelta + stateLh;
			st->lh = newLh;

		}

		//状态剪枝
		//b->prune(factory);

		//处理状态间跳转		
		StateTrace* previousBest = bestPreviousTrace(b);
		if (previousBest == NULL)
			continue;

		StateTrace* copy = factory->getInstance();
		copy->set(previousBest->enterTime, previousBest->cbidx, previousBest->lh, 0, previousBest->prev, -1);
		if (previousBest->prev != NULL)
			previousBest->prev->refcnt++;

		double durLh = bc->getDurLh(cbidx, 1);
		double newLh = durLh + stateLh + previousBest->lh;
		StateTrace* newTrace = factory->getInstance();
		newTrace->set(time, cbidx, newLh, 0, copy, -1);
		copy->refcnt++;

		b->addStateTrace(newTrace);

		//状态剪枝
		prune(b);
	}
}

SegmentAlgorithm::~SegmentAlgorithm() {
	if (ansList != NULL)
		delete [] ansList;
}