#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../SpeechSegmentAlgorithm/SimpleSTFactory.h"
#include "CmdWordRecg.h"
#include "../CommonLib/Feature/20dBEnergyGeometryAverageMFCC45.h"
using namespace std;

CmdWordRecg::~CmdWordRecg() {
	delete u.dict;
	delete u.factory;
	delete u.bc;
	delete set;
	delete allCmdMask;
}

bool CmdWordRecg::isWord(char c) {
	return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

void CmdWordRecg::readCmdBuf(const char* buf, int n) {

	int p = 0;
	char py[20];
	char label[20];

	vector<int> tvec;
	int cmdIdx = 0;


	while (p < n) {
		while (p < n && !isWord(buf[p]))
			p++;
		if (p == n)
			break;
		
		int q = p;
		while (q < n && isWord(buf[q]) && q - p < 10)
			q++;
		string t(buf + p, buf + q);
		tvec.push_back(u.dict->textToWord(t));

		p = q;
		if (p == n)
			break;

		if (!isWord(buf[p]) && buf[p] != ' ') {
			sprintf(label, "word%d", cmdIdx);
			string tl(label);
			cmdLabel.push_back(tl);

			cmdset.push_back(tvec);
			tvec.clear();

			cmdIdx++;
		}
	}
	if (tvec.size() > 0) {
		sprintf(label, "word%d", cmdIdx);
		string tl(label);
		cmdLabel.push_back(tl);

		cmdset.push_back(tvec);
	}

}


CmdWordRecg::CmdWordRecg(const char* cmdWordFile, const char* codebookFile, const char* dictFile, double durWeight, bool useCuda, bool useSegmemtModel) {

	this->useCuda = useCuda;
	this->useSegmentModel = useSegmemtModel;

	set = new GMMCodebookSet(codebookFile, GMMCodebookSet::INIT_MODE_FILE);

	//printf("Mixnum = %d\n", set->getMixNum());

	GMMProbBatchCalc* gbc =	new GMMProbBatchCalc(set, useCuda, useSegmentModel);
	gbc->setDurWeight(durWeight);
	u.bc = gbc;

	//string dictPath(dictFile);
	u.dict = new WordDict(dictFile);

	SimpleSTFactory* stfact = new SimpleSTFactory();
	u.factory = stfact;

	ifstream infile;
	infile.open(cmdWordFile);
	if (!infile) {
		printf("cannot open cmd word file [%s]\n", cmdWordFile);
		exit(-1);
	}

	string line;
	while (getline(infile, line)) {
		vector<int> tvec;
		istringstream iss(line);

		string t;
		iss >> t;
		cmdLabel.push_back(t);

		while (iss >> t) {
			tvec.push_back(u.dict->textToWord(t));
		}
		cmdset.push_back(tvec);
	}
	infile.close();

	int cbNum = u.dict->getTotalCbNum();
	allCmdMask = new bool[cbNum];
	for (int i = 0; i < cbNum; i++) {
		allCmdMask[i] = false;
	}
	for (int i = 0; i < cmdset.size(); i++) {
		int ansNum = cmdset[i].size();
		int* ansList = new int[ansNum];
		for (int j = 0; j < ansNum; j++) {
			ansList[j] = cmdset[i][j];
		}

		bool* mask = new bool[cbNum];
		u.dict->getUsedStateIdInAns(mask, ansList, ansNum);

		for (int j = 0; j < cbNum; j++) {
			if (mask[j]) {
				allCmdMask[j] = true;
			}
		}
		delete [] mask;
		delete [] ansList;
	}
	
}

bool CmdWordRecg::errorLessThan(const ResPair& m1, const ResPair& m2) {
	return m1.second.lh > m2.second.lh;	//lh1 > lh2 means error1 < error2, here error = -lh
}


string CmdWordRecg::cmdIdxToLabel(int idx) {
	return cmdLabel.at(idx);
}


int CmdWordRecg::cmdRecgIdx(short* sample, int sampleNum) {
	CmdRecgResult res = cmdRecg(sample, sampleNum);
	if (res.rank.size() == 0) {
		return -1;
	}
	return res.rank[0];

}

CmdRecgResult CmdWordRecg::cmdRecg(short* samples, int sampleNum) {
	//FrameNum = (SpeechSampleNum-FRAME_LEN+FRAME_STEP)/FRAME_STEP;
	int frameNum = (sampleNum - FRAME_LEN + FRAME_STEP) / FRAME_STEP;
	auto featureBuf = (float(*)[DIM])malloc(frameNum * DIM * sizeof(float));
	if (featureBuf == NULL ) {
		printf("cannot malloc memory for FeatureBuf\n");
		exit(-1);
	}
	get20dBEnergyGeometryAveragMfcc(samples, featureBuf, frameNum);
	

	SegmentAlgorithm sa;
	int cmdNum = cmdset.size();
	if (cmdNum == 0) {
		printf("cmd word set is empty\n");
		exit(-1);
	}

	vector<ResPair> allres;
	float* features = (float*)featureBuf;

	//FILE* shit = fopen("mtf.fm","wb");
	//fwrite(features,sizeof(float),frameNum,shit);
	//fclose(shit);

	GMMProbBatchCalc* gbc = (GMMProbBatchCalc*)u.bc;
	gbc->setMask(allCmdMask);

	double* fd = new double[frameNum * DIM];
	for (int j = 0; j < frameNum * DIM; j++) {
		fd[j] = features[j];
	}

	gbc->prepare(fd, frameNum);



	for (int i = 0; i < cmdNum; i++) {

		int ansNum = cmdset.at(i).size();
		int* ansList = new int[ansNum];
		for (int j = 0; j < ansNum; j++) {
			ansList[j] = cmdset.at(i).at(j);
		}

		SegmentResult t = sa.segmentSpeech(frameNum, DIM, ansNum, ansList, u);
		if (t.frameLabel.size() > 0) 
			allres.push_back(std::make_pair(i, t));

		delete [] ansList;
	}
	delete [] fd;

	free(featureBuf);
	CmdRecgResult r;
	if (allres.size() > 0) {
		std::sort(allres.begin(), allres.end(), errorLessThan);
		for (auto i = allres.begin(); i != allres.end(); i++) {
			r.rank.push_back(i->first);
			r.lh.push_back(i->second.lh);
			r.text.push_back(cmdLabel[i->first]);
		}
	}

	return r;
}