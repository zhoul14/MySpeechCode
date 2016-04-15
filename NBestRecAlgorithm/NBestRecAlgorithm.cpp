#include "NBestRecAlgorithm.h"
//#include "boost/pool/pool.hpp"
#include <unordered_set>
#include <deque>
#include <assert.h>
#include <queue> 
#include <unordered_map>
#include "omp.h"


int NBestRecAlgorithm::tokenFilterStep(TokenBin* bin, int maxCandNum, RecToken** tokenCand) {

			
	int cbType = dict->getCbType(bin->cbidx);
	bool distinctWordIdCheck = bin->isEndingBin || isCrossWordCb(cbType);
	//是否要求候选token的WordId各不相同
	//bool distinctCIdCheck = !distinctWordIdCheck && isInVowel(cbType);	//是否要求候选token的CId各不相同


	bool sylSelected[TOTAL_SYLLABLE_NUM];
	if (distinctWordIdCheck) {
		for (int i = 0; i < TOTAL_SYLLABLE_NUM; i++) {
			sylSelected[i] = false;
		}
	}

// 	bool CIdSelected[INITIAL_WORD_NUM];
// 	if (distinctCIdCheck) {
// 		for (int i = 0; i < INITIAL_WORD_NUM; i++) {
// 			CIdSelected[i] = false;
// 		}
// 	}

	std::vector<TokenBin*>& previous = bin->previous;
	int n = previous.size();

	int* ptrs = new int[n];
	int activeCnt = 0;
	for (int i = 0; i < n; i++) {
		if (previous[i]->tokenNum() == 0) {
			ptrs[i] = -1;
		} else {
			ptrs[i] = 0;
			activeCnt++;
		}
	}

	bool noiseSelected = false;
	int selectedCnt = 0;
	RecToken* allBestToken = NULL;

	while (selectedCnt < maxCandNum && activeCnt > 0) {
		int maxBinIdx = -1;
		double maxBinLh = 0;
		RecToken* maxToken = NULL;
		for (int i = 0; i < n; i++) {
			if (ptrs[i] < 0)
				continue;

			RecToken* tmpToken = previous[i]->content[ptrs[i]];
			double tmpLh = tmpToken->lh;
			if (maxBinIdx == -1 || tmpLh > maxBinLh) {
				maxBinIdx = i;
				maxBinLh = tmpLh;
				maxToken = tmpToken;
			}
		}

		if (distinctWordIdCheck) {//在字边界处要求候选token对应的Word各不相同
			if (maxToken->wordId == -1) {
				if (!noiseSelected) {
					tokenCand[selectedCnt++] = maxToken;
					noiseSelected = true;
				}
			} else {
				int sylId = dict->wordToSyllable(maxToken->wordId);
				if (!sylSelected[sylId]) {
					tokenCand[selectedCnt++] = maxToken;
					sylSelected[sylId] = true;
				}
			}
		}
// 		else if (distinctCIdCheck) {
// 			int cid = maxToken->CId;
// 			if (!CIdSelected[cid]) {
// 				tokenCand[selectedCnt++] = maxToken;
// 				CIdSelected[cid] = true;
// 			}
// 		}
		else {
			tokenCand[selectedCnt++] = maxToken;
		}
		
		//FINAL0 对应着C-V跳转，决定了token的WordId，令FINAL0尽量来自不同的码本以提高WordId的多样性
		if (bin->cbidx >= 0 && cbType == FINAL0) {	
			ptrs[maxBinIdx] = -1;
			activeCnt--;
		} else {
			TokenBin* bestBin = previous[maxBinIdx];
			ptrs[maxBinIdx]++;
			if (ptrs[maxBinIdx] >= bestBin->tokenNum()) {
				ptrs[maxBinIdx] = -1;
				activeCnt--;
			}

		}
	}

	delete [] ptrs;

	return selectedCnt;

}

int NBestRecAlgorithm::funcCallCnt = 0;

//candToken中的候选字也已经按似然值降序排好序，则返回值中的元素已经按照似然值的降序排好序
int NBestRecAlgorithm::candMergeStep(TokenBin* bin, RecToken** candToken,
 	int candTokenNum, int maxCandNum, WordNode** resWord, double* resLhDiff) {

	int selectedCand = 0;
	if (candTokenNum == 0) {
		return 0;
	}

	bool crossWord = true;
	if (bin->cbidx >= 0) {
		int cbType = dict->getCbType(bin->cbidx);
		crossWord = isCrossWordCb(cbType);
	}

	//如果是字内跳转
	if (!crossWord) {
		if (candTokenNum == 1) {
			RecToken* lastToken = candToken[0];
			for (int i = 0; i < lastToken->candWordNum(); i++) {
				if (selectedCand == maxCandNum)
					return selectedCand;

				resWord[i] = lastToken->candWord[i];
				resLhDiff[i] = lastToken->lhDiff[i];
				selectedCand++;
			}
			return selectedCand;
		} else {
			int* candPtr = new int[candTokenNum];
			int activeBinCnt = 0;
			for (int i = 0; i < candTokenNum; i++) {
				if (candToken[i]->candWordNum() == 0) {
					candPtr[i] = -1;
				} else {
					candPtr[i] = 0;
					activeBinCnt++;
				}
			}

			bool sylSelected[TOTAL_SYLLABLE_NUM];
			for (int i = 0; i < TOTAL_SYLLABLE_NUM; i++) {
				sylSelected[i] = false;
			}
			bool noiseSelected = false;
			double bestLh = candToken[0]->lh;

 			while (selectedCand < maxCandNum && activeBinCnt > 0) {

				double maxLh = 0;
				int maxTokenIdx = -1;
				RecToken* maxToken = NULL;

				for (int i = 0; i < candTokenNum; i++) {
					if (candPtr[i] < 0)
						continue;
					RecToken* t = candToken[i];

					double tmpLh = t->lh + t->lhDiff[candPtr[i]];
					if (maxTokenIdx == -1 || tmpLh > maxLh) {
						maxLh = tmpLh;
						maxTokenIdx = i;
						maxToken = t;
					}
				}
				
				int candWordId = maxToken->candWord[candPtr[maxTokenIdx]]->wordId;
				int sylId = candWordId == -1 ? -1 : dict->wordToSyllable(candWordId);

				bool candIsNoise = (candWordId == -1);
				bool nSelected = noiseSelected;
				bool sSelected = candIsNoise || sylSelected[sylId];
				/*bool tokenFromSameWord = candToken[0]->wordId == maxToken->wordId;*/
			
				bool newValidWord = (candIsNoise && !nSelected) || (!candIsNoise && !sSelected);
				sylSelected[sylId] |= !candIsNoise;
				noiseSelected = nSelected | candIsNoise;

				if (newValidWord) {
					WordNode* w = maxToken->candWord[candPtr[maxTokenIdx]];
					double newDiff = maxLh - bestLh;
					resWord[selectedCand] = w;
					resLhDiff[selectedCand] =  newDiff;
					selectedCand++;
				}

				int nextCandPtr = candPtr[maxTokenIdx] + 1;
				if (nextCandPtr >= maxToken->candWordNum()) {
					activeBinCnt--;
					candPtr[maxTokenIdx] = -1;
				} else
					candPtr[maxTokenIdx] = nextCandPtr;
 			}
			delete [] candPtr;
		}
	} else {	//如果是字间跳转
		double bestLh = candToken[0]->lh;
		for (int i = 0; i < candTokenNum; i++) {
			WordNode* w;

			w = fact.getInstance();		//所有PreviousWord的创建都来自这个分支

			w->wordId = candToken[i]->wordId;
			w->endTime = time;
			w->endLh = candToken[i]->lh;
			w->candTrimmed = false;
			for (int j = 0; j < 7; j++) {
				w->cbTrace[j] = candToken[i]->cbTrace[j];
				w->jumpTime[j] = candToken[i]->jumpTime[j];
				w->lhTrace[j] = candToken[i]->lhTrace[j];
			}
			//w->lhTrace[7] = w->endLh;


			for (int j = 0; j < candToken[i]->candWordNum(); j++) {
				WordNode* wt = candToken[i]->candWord[j];
				double dt = candToken[i]->lhDiff[j];
				w->addPrev(wt, dt);
			}

			double lhDiff = candToken[i]->lh - bestLh;

			resWord[selectedCand] = w;
			resLhDiff[selectedCand] = lhDiff;

			selectedCand++;
			if (selectedCand >= maxCandNum)
				break;
		}
		

	}

 	return selectedCand;

} 


void NBestRecAlgorithm::constructBinSet() {
	if (binSet.size() != 0) {
		std::cout << "binSet.size() != 0, the recognizer is not in a clear state";
		exit(-1);
	}

	for (int i = 0; i < cbNum; i++) {
		TokenBin* bin = new TokenBin(i);
		bin->isEndingBin = false;
		binSet.push_back(bin);
	}

	for (int i = 0; i < cbNum; i++) {
		std::vector<int> jmpList = dict->getJumpList(i);
		for (int j = 0; j < jmpList.size(); j++) {
			if(jmpList.at(j)>=cbNum)
				continue;
			TokenBin* prev = binSet.at(jmpList.at(j));
			binSet.at(i)->addPreviousBin(prev);
		}
	}

	TokenBin* endingBin = new TokenBin(-1);
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
			cbidx = dict->getDiIsoCbId(i, DI_FINAL3);
		}
		endingBin->addPreviousBin(binSet.at(cbidx));
	}

	endingBin->addPreviousBin(binSet.at(dict->getNoiseCbId()));

	binSet.push_back(endingBin);


}

void NBestRecAlgorithm::resetBinSet() {
	for (auto i = binSet.begin(); i != binSet.end(); i++) {
		std::vector<RecToken*>* stlist = &((*i)->content);
		for (auto j = stlist->begin(); j != stlist->end(); j++) {
			(*j)->releaseCand(&fact);
			delete *j;
		}
	}
}

void NBestRecAlgorithm::clearBinSet() {
	for (int i = 0; i < binSet.size(); i++) {
		std::vector<RecToken*>* stlist = &(binSet.at(i)->content);
		for (int j = 0; j < stlist->size(); j++) {
			RecToken* t = stlist->at(j);
			t->releaseCand(&fact);
			delete t;
		}
		stlist->clear();
	}
}



void NBestRecAlgorithm::releaseBinSet() {
	for (auto i = binSet.begin(); i != binSet.end(); i++) {
		std::vector<RecToken*>* stlist = &((*i)->content);
		for (auto j = stlist->begin(); j != stlist->end(); j++) {
			(*j)->releaseCand(&fact);
			delete *j;
		}
		delete *i;
	}
	binSet.clear();
}

std::vector<std::vector<SWord> > NBestRecAlgorithm::genAlignedResult(WordNode* l) {
	if (l->prevNum() == 0) {
		printf("empty lattice in genRecResult!\n");
		exit(-1);
	}

	std::vector<std::vector<SWord> > res;
	WordNode* p = l;

	while (1) {
		if (p->prevNum() == 0)
			break;
		if (p->prev[0]->wordId == -1) {
			p = p->prev[0];
			continue;
		}

		std::vector<SWord> t;
		for (int i = 0; i < p->prevNum(); i++) {
			SWord r;
			r.wordId = p->prev[i]->wordId;
			r.lh = p->prevLhDiff[i] + p->prev[0]->endLh;
			memcpy(r.jumpTime,p->prev[i]->jumpTime, 7 * sizeof(int));
			r.endTime = p->prev[i]->endTime;
			t.push_back(r);
		}
		res.push_back(t);

		p = p->prev[0];
	}

	int N = res.size();
	std::vector<std::vector<SWord> > res2;
	for (int i = 0; i < N; i++) {
		int k = N - i - 1;
		res2.push_back(res.at(k));
	}

	for (int i = 0; i < res2.size(); i++) {
		for (int j = 0; j < res2[i].size(); j++) {
			res2[i][j].label = dict->wordToText(res2[i][j].wordId);
		}
	}
	return res2;

}

void NBestRecAlgorithm::getJumpFlag(WordNode* x, int* flag, int fnum) {
	for (int i = 0; i < fnum; i++)
		flag[i] = LIMITED_JUMP_DISALLOW;
	WordNode* p = x;
	while (p->prev[0] != NULL) {
		p = p->prev[0];
		int time = p->endTime;
		if (time != fnum - 1 && time >= 0) {
			flag[time] = LIMITED_JUMP_ALLOW;
		}
	}
	return;
}



std::vector<std::vector<SWord> > NBestRecAlgorithm::recSpeech(int fnum, int fDim, const WordDict* dict, GMMProbBatchCalc* bc, int multiJump, bool useSegmentModel) {

	if (multiJump <= 0 || multiJump > BEST_N) {
		multiJump = BEST_N;
	}
	VJumpLimit = multiJump;

	int* cwJump = new int[fnum];
	for (int i = 0; i < fnum; i++)
		cwJump[i] = FREE_JUMP;//FREE_JUMP

	WordNode* x = recSpeech(fnum, fDim, dict, bc, cwJump, useSegmentModel);
	getJumpFlag(x, cwJump, fnum);

	fact.destroyInstance(x);
	fact.checkClear();
	
	x = recSpeech(fnum, fDim, dict, bc, cwJump, useSegmentModel);
	std::vector<std::vector<SWord> > res = genAlignedResult(x);

	fact.destroyInstance(x);
	fact.checkClear();
	delete [] cwJump;
	return res;
}

WordNode* NBestRecAlgorithm::recSpeech(int fnum, int fDim, const WordDict* dict, GMMProbBatchCalc* bc, int* allowCWJump, bool useSegmentModel) {
	this->bc = bc;
	this->fDim = fDim;
	this->dict = dict;
	this->cbNum = dict->getTotalCbNum();
	this->useSegmentModel = useSegmentModel;
	tokenCnt = 0;

	startNoiseNode = fact.getInstance();
	startNoiseNode->wordId = -1;
	startNoiseNode->endLh = 0;
	startNoiseNode->endTime = -1;
	
	time = -1;

	constructBinSet();

	for (int i = 0; i < fnum; i++) {
		frameSegmentation(allowCWJump[i]);
		//printf("frameSegmentation %d/%d\n", i, fnum);
	}

	//最后一次跳转，收集结果
	TokenBin* endingBin = binSet.at(cbNum);
	RecToken* candToken[BEST_N];

	int candTokenNum = tokenFilterStep(endingBin, BEST_N, candToken);
	WordNode* resw = fact.getInstance();
	if (candToken[0]->wordId == -1) {

		for (int i = 0; i < candToken[0]->candWordNum(); i++) {
			if (candToken[0]->candWord[i] != NULL) {
				resw->addPrev(candToken[0]->candWord[i], candToken[0]->lhDiff[i]);
			}
		}

		resw->endLh = candToken[0]->candWord[0]->endLh;

	} else {

		WordNode* newCandWords[BEST_N];
		double lhDiff[BEST_N];

		int preWordNum = candMergeStep(endingBin, candToken, candTokenNum, BEST_N, newCandWords, lhDiff);

		for (int i = 0; i < preWordNum; i++) {
			resw->addPrev(newCandWords[i], lhDiff[i]);
		}
	}

	releaseBinSet();
	return resw;

}

std::vector<std::vector<LatticeNode> > NBestRecAlgorithm::rearrangeLattice(int fnum, WordNode* lastNode) {
	using namespace std;
	vector<vector<LatticeNode> > res;
	res.resize(fnum);

	vector<vector<WordNode*> > rest;
	rest.resize(fnum);

	deque<WordNode*> q;
	if (lastNode->prevNum() == 0)
		return res;

	unordered_set<WordNode*> foundNode;
	for (int i = 0; i < lastNode->prevNum(); i++) {
		WordNode* tn = lastNode->prev[i];
		q.push_back(tn);
		foundNode.insert(tn);
	}

	while (q.size() > 0) {

		WordNode* t = q.front();
		q.pop_front();

		if (t->endTime < 0)
			continue;

		rest[t->endTime].push_back(t);

		for (int i = 0; i < t->prevNum(); i++) {
			WordNode* tn = t->prev[i];
			auto prevPtr = foundNode.find(tn);
			if (prevPtr == foundNode.end()) {
				q.push_back(tn);
				foundNode.insert(tn);
			}
		}
	}

	for (int i = 0; i < fnum; i++) {
		vector<WordNode*>& tvec = rest[i];
		sort(tvec.begin(), tvec.end(), wordNodeLargerThan);
		for (int j = 0; j < tvec.size(); j++)
			tvec[j]->lhRank = j;
	}

	for (int i = 0; i < fnum; i++) {
		int L = rest[i].size();
		res[i].resize(L);
		for (int j = 0; j < L; j++) {
			WordNode* t = rest[i][j];
			LatticeNode ln;
			ln.endLh = t->endLh;
			ln.id.x = i;
			ln.id.y = j;

			for (int k = 0; k < t->prevNum(); k++) {
				
				NID prevNID;
				if (t->prev[k]->endTime < 0) {
					prevNID.x = -1;
					prevNID.y = -1;
				} else {
					prevNID.x = t->prev[k]->endTime;
					prevNID.y = t->prev[k]->lhRank;
				}
				

				ln.prevLhDiff.push_back(t->prevLhDiff[k]);
				ln.prev.push_back(prevNID);
			}
		

			ln.wordId = t->wordId;
			res[i][j] = ln;
		}
	}
	return res;
}

void NBestRecAlgorithm::timeZeroInit() {

	//在全部非协同发音I0状态中放入token
	for (int i = 0; i < INITIAL_WORD_NUM; i++) {
		int cbidx = dict->getIsoCbId(i, INITIAL0);
		double durLh = bc->getDurLh(cbidx, 1);
		double stateLh = bc->getStateLh(cbidx, time);

		RecToken* token = new RecToken();
		tokenCnt++;
		token->lh = durLh + stateLh;
		token->wordId = -1;
		token->currentCbId = cbidx;
		token->dur = 1;
		token->addPrev(startNoiseNode, 0);

		binSet.at(cbidx)->addToken(token);
	}

	//在全部协同发音I0状态中放入token
	if(dict->triPhone)
		for (int i = 0; i < C_CLASS_NUM; i++) {
			for (int j = 0; j < V_CLASS_NUM; j++) {
				int cbidx = dict->getCACbId(i, j, INITIAL0_C);
				double durLh = bc->getDurLh(cbidx, 1);
				double stateLh = bc->getStateLh(cbidx, time);

				RecToken* token = new RecToken();
				tokenCnt++;
				token->lh = durLh + stateLh;
				token->wordId = -1;
				token->currentCbId = cbidx;
				token->dur = 1;
				token->addPrev(startNoiseNode, 0);

				binSet.at(cbidx)->addToken(token);
			}
		}

	//在噪声状态中放入token
	int noiseIdx = dict->getNoiseCbId();
	double durLh = bc->getDurLh(noiseIdx, 1);
	double stateLh = bc->getStateLh(noiseIdx, time);

	RecToken* token = new RecToken();
	tokenCnt++;
	token->lh = durLh + stateLh;
	token->wordId = -1;
	token->currentCbId = noiseIdx;
	token->dur = 1;

	binSet.at(noiseIdx)->addToken(token);

	return;
}

int NBestRecAlgorithm::createNewToken(int cbidx, int cbType, RecToken** candToken,
	int candTokenNum, WordNode** newCandWords, double* lhDiff, 
	int preWordNum,	RecToken* (*tokenBuffer)[BEST_N], int* newTokenCnt) {

	if (candTokenNum <= 0)
		return 0;

	

	bool isCrossWord = isCrossWordCb(cbType);
	bool inVowel = isInVowel(cbType);
	bool inCoaVowel = dict->triPhone? cbType == FINAL2_C || cbType == FINAL3_C :false;

	double durLh = bc->getDurLh(cbidx, 1);
	double stateLh = bc->getStateLh(cbidx, time);

	bool CIdSelected[INITIAL_WORD_NUM];
	for (int i = 0; i < INITIAL_WORD_NUM; i++)
		CIdSelected[i] = false;
	
// 	bool VIdSelected[FINAL_WORD_NUM];
// 	for (int i = 0; i < FINAL_WORD_NUM; i++)
// 		VIdSelected[i] = false;

	newTokenCnt[cbidx] = 0;

	for (int i = 0; i < candTokenNum; i++) {

		
		if (inVowel) {
// 			if (inCoaVowel) {
// 				int candCId = candToken[i]->CId;
// 				int candVId = candToken[i]->VId;
// 				if (CIdSelected[candCId] && VIdSelected[candVId])
// 					continue;
// 				CIdSelected[candCId] = true;
// 				VIdSelected[candVId] = true;
// 			} else {
				int candCId = candToken[i]->CId;
				if (CIdSelected[candCId])
					continue;
				CIdSelected[candCId] = true;
//			}
			
		}

		RecToken* t = new RecToken();
		tokenCnt++;
		t->lh = candToken[i]->lh + stateLh + durLh;
		t->dur = 1;

		if (!isCrossWord) {
			for (int j = 0; j <= HMM_STATE_DUR; j++) {
				t->jumpTime[j] = candToken[i]->jumpTime[j];
				t->cbTrace[j] = candToken[i]->cbTrace[j];
				t->lhTrace[j] = candToken[i]->lhTrace[j];
			}
		}

		t->wordId = candToken[i]->wordId;
		t->CId = candToken[i]->CId;
		t->VId = candToken[i]->VId;
		
		if (isCrossWord) {
			t->wordId = -1;
			t->CId = RecToken::UNINITIALIZED_CID;
			t->VId = RecToken::UNINITIALIZED_VID;
		} else if (cbType == INITIAL1 && dict->triPhone || cbType == DI_INITIAL1 && !dict->triPhone) {
			t->CId = dict->getCVIdFromCbId(cbidx);
		} else if (cbType == FINAL0 && dict->triPhone || cbType == DI_FINAL0 && !dict->triPhone) {
			t->VId = dict->getCVIdFromCbId(cbidx);
			t->wordId = dict->getWordIdFromCVLink(candToken[i]->currentCbId, cbidx);
		}

		

		for (int j = 0; j < preWordNum; j++) {
			t->addPrev(newCandWords[j], lhDiff[j]);
		}

		if (isCrossWord) {
			for (int j = 0; j < preWordNum; j++) {
				fact.unkeyPrune(t->candWord[j]);
			}
		}

		t->setJumpTimeAndCb(cbType, cbidx, time, t->lh, dict->triPhone);

		t->currentCbId = cbidx;
		tokenBuffer[cbidx][newTokenCnt[cbidx]] = t;
		newTokenCnt[cbidx]++;

		if (!inVowel)
			break;

		if (newTokenCnt[cbidx] >= VJumpLimit)
			break;
	}

	return newTokenCnt[cbidx];
}

void NBestRecAlgorithm::frameSegmentation(int allowCrossWordJump) {
	time++;

	if (time == 0) {
		timeZeroInit();
		return;
	}

	RecToken* (*tokenBuffer)[BEST_N] = new RecToken*[cbNum][BEST_N];
	for (int i = 0; i < cbNum; i++) {
		for (int j = 0; j < BEST_N; j++) {
			tokenBuffer[i][j] = NULL;
		}
	}

	int* newTokenCnt = new int[cbNum];
	memset(newTokenCnt, 0, cbNum * sizeof(int));

	int* cbTypeLookup = new int[cbNum];
	for (int i = 0; i < cbNum; i++) {
		cbTypeLookup[i] = dict->getCbType(i);
	}

	omp_set_dynamic(true);
	#pragma omp parallel for 
	for (int i = 0; i < cbNum; i++) {
		TokenBin* bin = binSet[i];
		RecToken* candToken[BEST_N];

		WordNode* newCandWords[BEST_N];
		double lhDiff[BEST_N];

		for (int j = 0; j < BEST_N; j++) {
			candToken[j] = NULL;
			newCandWords[j] = NULL;
			lhDiff[j] = 0;
		}

		int cbType = cbTypeLookup[i];
		bool isCrossWord = isCrossWordCb(cbType);
		bool noJump = allowCrossWordJump == LIMITED_JUMP_DISALLOW && isCrossWord ||
			allowCrossWordJump == LIMITED_JUMP_ALLOW && !isCrossWord;
		if (noJump)
			continue;

		int candTokenNum = tokenFilterStep(bin, BEST_N , candToken);
		int preWordNum = candMergeStep(bin, candToken, candTokenNum, BEST_N, newCandWords, lhDiff);
		int newTokenNum = createNewToken(i, cbType, candToken, candTokenNum, newCandWords,
			lhDiff, preWordNum, tokenBuffer, newTokenCnt);
	}

	//状态驻留
	for (int i = 0; i < cbNum; i++) {
		TokenBin* bin = binSet.at(i);
		double stateLh = bc->getStateLh(i, time);

		for (auto j = bin->content.begin(); j != bin->content.end(); j++) {
			RecToken* t0 = *j;

			t0->dur += 1;
			int dur = t0->dur;
			double deltaDurLh = bc->getDurLh(i, dur) - bc->getDurLh(i, dur - 1);

			t0->lh = t0->lh + deltaDurLh + stateLh;
		}
	}

	if (allowCrossWordJump == LIMITED_JUMP_ALLOW) {
		clearBinSet();
	}

	//完成状态跳转
	for (int i = 0; i < cbNum; i++) {

		TokenBin* bin = binSet.at(i);
		for (int j = 0; j < newTokenCnt[i]; j++)
			if (tokenBuffer[i][j] != NULL)
				bin->addToken(tokenBuffer[i][j]);
		prune(bin);
	}

	delete [] cbTypeLookup;
	delete [] newTokenCnt;
	delete [] tokenBuffer;
}

void NBestRecAlgorithm::prune(TokenBin* bin) {
	std::vector<RecToken*>& list = bin->content;
	if (list.size() < 2)
		return;
	if (useSegmentModel) {
		int currPos = list.size() - 1;
		while (currPos > 0) {
			RecToken* currToken = list[currPos];
			RecToken* prevToken = list[currPos - 1];

			if (prevToken->lh < currToken->lh) {

				bool differentSyl = currToken->wordId >= 0 && 
					prevToken->wordId >= 0 && 
					dict->wordToSyllable(currToken->wordId) != dict->wordToSyllable(prevToken->wordId);
				if (list.size() <= pruneThreshold && differentSyl) {
					int insertPos = currPos;
					while (insertPos != list.size() && list[insertPos]->lh > prevToken->lh)
						insertPos++;
					list.insert(list.begin() + insertPos, prevToken);
				} else {
					prevToken->releaseCand(&fact);
					delete prevToken;
					tokenCnt--;
				}

				list.erase(list.begin() + currPos - 1);
			}
			currPos--;
		}
	} else {
		int cbType = dict->getCbType(bin->cbidx);
 		if (isInInitial(cbType)) {
 			std::sort(list.begin(), list.end(), RecToken::tokenCmp);
 			int L = list.size();
 			int n = L - pruneThreshold;
			
			for (int i = 0; i < n; i++) {
				list[L - 1 - i]->releaseCand(&fact);
				delete list[L - 1 - i];
				tokenCnt--;
			}

			for (int i = 0; i < n; i++) {
				list.pop_back();
			} 
		} else {

			int L = list.size();
			std::unordered_map<int, int> sylMap;
			for (int i = 0; i < L; i++) {
				int syl = dict->wordToSyllable(list[i]->wordId);
				auto searchRes = sylMap.find(syl);
				if (searchRes != sylMap.end()) {
					double p = searchRes->second;
					if (list[i]->lh > list[p]->lh) {
						sylMap[syl] = i;
					}
				} else {
					sylMap[syl] = i;
				}
			}

			std::vector<RecToken*> tmpVec;
			bool* selected = new bool[L];
			for (int i = 0; i < L; i++) {
				selected[i] = false;
			}
			for (auto i = sylMap.begin(); i != sylMap.end(); i++) {
				int idx = i->second;
				tmpVec.push_back(list[idx]);
				selected[idx] = true;
			}

			for (int i = 0; i < list.size(); i++) {
				if (!selected[i]) {
					list[i]->releaseCand(&fact);
					delete list[i];
					tokenCnt--;
				}
			}
			delete [] selected;
			list.clear();

			std::sort(tmpVec.begin(), tmpVec.end(), RecToken::tokenCmp);
			int n = tmpVec.size();
			if (n > pruneThreshold)
				n = pruneThreshold;
			for (int i = 0; i < n; i++) {
				list.push_back(tmpVec[i]);
			}

			for (int i = n; i < tmpVec.size(); i++) {
				tmpVec[i]->releaseCand(&fact);
				delete tmpVec[i];
				tokenCnt--;
			}

			

		}
		
	}
	
}
