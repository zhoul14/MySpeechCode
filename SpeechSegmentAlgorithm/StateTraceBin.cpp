#include "StateTraceBin.h"

void StateTraceBin::addStateTrace(StateTrace* s) {
	if (s == NULL) {
		std::cout << "state trace pointer is null";
		exit(-1);
	}
	content.push_back(s);
}

void StateTraceBin::addPreviousBin(StateTraceBin* b) {
	if (b == NULL) {
		std::cout << "state trace bin pointer is null";
		exit(-1);
	}
	previous.push_back(b);
}

StateTraceBin::StateTraceBin(int cbidx) {
	this->cbidx = cbidx;
}

StateTrace* StateTraceBin::getBestTrace() {
	return content.size() == 0 ? NULL : content.front();
}



std::vector<StateTrace*> StateTraceBin::getPreviousBest(int maxStNum, bool distinctSylCheck, WordDict* dict) {
	std::vector<StateTrace*> stateCand;

	bool* sylSelected = NULL;
	if (distinctSylCheck) {
		sylSelected = new bool[TOTAL_SYLLABLE_NUM];
		for (int i = 0; i < TOTAL_SYLLABLE_NUM; i++) {
			sylSelected[i] = false;
		}
	}

	int activeBinCnt = 0;

	int n = previous.size();
	int* ptrs = new int[n];

	for (int i = 0; i < n; i++) {
		if (previous.at(i)->stateTraceNum() == 0) {
			ptrs[i] = -1;
		} else {
			ptrs[i] = 0;
			activeBinCnt++;
		}
	}

	int selectedCnt = 0;
	while (selectedCnt < maxStNum) {
		if (activeBinCnt == 0) {
			break;
		}

		int maxBinIdx = -1;
		double maxBinLh = 0;
		StateTrace* maxStateTrace = NULL;

		for (int i = 0; i < n; i++) {
			if (ptrs[i] < 0)
				continue;

			StateTrace* t = previous.at(i)->content.at(ptrs[i]);
			if (maxBinIdx == -1 || t->getLikelihood() > maxBinLh) {
				maxBinIdx = i;
				maxBinLh = t->getLikelihood();
				maxStateTrace = t;
			}
		}

		if (distinctSylCheck) {
			int sylId = dict->wordToSyllable(maxStateTrace->getWordId());
			if (!sylSelected[sylId]) {
				stateCand.push_back(maxStateTrace);
				selectedCnt++;
				sylSelected[sylId] = true;
			}
		} else {
			stateCand.push_back(maxStateTrace);
			selectedCnt++;
		}

		int t = ptrs[maxBinIdx] + 1;
		if (t == previous.at(maxBinIdx)->stateTraceNum()) {
			ptrs[maxBinIdx] = -1;
			activeBinCnt--;
		} else {
			ptrs[maxBinIdx]++;
		}

	}

	delete [] ptrs;
	if (distinctSylCheck) {
		delete [] sylSelected;
	}

	if (stateCand.size() > maxStNum) {
		printf("error in prepareJumpState, stateCand.size() = %d, maxStNum = %d\n", stateCand.size(), maxStNum);
		exit(-1);
	}

	return stateCand;
}

int StateTraceBin::stateTraceNum() {
	return content.size();
}