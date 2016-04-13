#include <stdio.h>
#include <vector>
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../GMMCodebook/GMMCodebookSet.h"
#include "../CommonLib/FileFormat/FeatureFileSet.h"
#include "../CommonLib/Dict/WordDict.h"
//#include "boost/filesystem.hpp"
#include "../CommonLib/ReadConfig/RecParam.h"
#include "../OneBestRecAlgorithm/SimpleSpeechRec.h"
//#include "SRecTokenFactory.h"
#include <windows.h>
using namespace std;
void printRecFile(FILE* fid, const vector<SWord>& res) {
	fprintf(fid, "%4d\n", res.size());
	for (int i = 0; i < res.size(); i++) {
		SWord r = res[i];
		if (r.wordId == -1)
			continue;
		fprintf(fid, "%4d %.2f ", r.wordId, r.lh);
	}
	fprintf(fid, "\n");
	fflush(fid);
}

void printRecResToScreen(const vector<SWord>& res, int* ansId, int ansNum, WordDict* dict) {
	if (res.size() == 0)
		return;

	for (int i = 0; i < res.size(); i++) {
		string tw = dict->wordToText(res[i].wordId);
		printf("%s[%.2f] ", tw.c_str(), res[i].lh);
	}
	printf("\n");
	
	int s = res.size();
	printf("LH = %f\n", res[s - 1].lh);

	printf("[T]");
	for (int i = 0; i < ansNum; i++) {
		string tw = dict->wordToText(ansId[i]);
		printf("%s ", tw.c_str());
	}
	printf("\n\n");

}


int main(int argc,char *argv[]) {

	char *recg;
	if(argc < 2 || argc > 3) {
		printf("usage:program_name config_file [basedir]\n");
		exit(-1);
	} else {
		recg = argv[1];
		if (argc == 3) {
			//boost::filesystem::current_path(argv[2]);
			SetCurrentDirectory(argv[2]);
		}
	}

	RecParam rparam(recg);
	std::string CodeBookName = rparam.getCodebookFileName();
	int recFileNum = rparam.getRecNum();
	int multiJump = rparam.getMultiJump();

	const int bestN = BEST_N;

	if (recFileNum < 1) {
		printf("Recognition File Number error! FileNum:%ld\n", recFileNum);
		return -1;
	}
	printf("FileNum=%ld\n", recFileNum);

	GMMCodebookSet* cbset = new GMMCodebookSet(rparam.getCodebookFileName().c_str());

	bool useSegmentModel = rparam.getSegmentModelFlag();

	bool useCuda = rparam.getUseCudaFlag();
	double Coef = rparam.getDCoef();
	GMMProbBatchCalc* gbc = new GMMProbBatchCalc(cbset, useCuda, useSegmentModel, Coef);

	WordDict* dict = new WordDict(rparam.getWordDictFileName().c_str(),rparam.getTriPhone());

	const int fDim = cbset->FDim;

	SimpleSpeechRec* reca = new SimpleSpeechRec();

	gbc->setDurWeight(rparam.getDurWeight());
	printf("CodeBook:%s  DurWeight=%f\n",CodeBookName.c_str(),rparam.getDurWeight());

	vector<RSpeechFile> inputs = rparam.getRecFiles();
	double totalLh = 0;

	for (int i = 0; i < recFileNum; i++) {

		RSpeechFile input = inputs.at(i);
		FeatureFileSet fs(input.getFeatureFileName(), input.getMaskFileName(), input.getAnswerFileName(), cbset->FDim);

		string pdir = input.getRecResultFileName();
		pdir = pdir.substr(0, pdir.find_last_of("/\\"));
		if (GetFileAttributes(pdir.c_str()) == INVALID_FILE_ATTRIBUTES) {
			CreateDirectory(pdir.c_str(), NULL);
		}

		FILE* recf = fopen(input.getRecResultFileName().c_str(), "w");
		if (!recf) {
			printf("Cannot open file: %s\n", input.getRecResultFileName().c_str());
			exit(-1);
		}


		//读入语音特征文件
		int SentenceNum = fs.getSpeechNum();
		fprintf(recf, "%4d\n", SentenceNum);

		for (int j = 0; j < SentenceNum ; j++) {

			printf("recognizing file %d: %d/%d\n", i, j, SentenceNum);
			int wordNum = fs.getWordNumInSpeech(j);
			int* pWordList = new int[wordNum];
			fs.getWordListInSpeech(j, pWordList);

			int fNum = fs.getFrameNumInSpeech(j);
			double* features = new double[fNum * fDim];
			fs.getSpeechAt(j, features);

			gbc->prepare(features, fNum);
			vector<SWord> res = reca->recSpeech(fNum, fDim, dict, gbc, useSegmentModel,rparam.getBHeadNoise());
			printRecResToScreen(res, pWordList, wordNum, dict);

			int resLen = res.size();
			totalLh += res[resLen - 1].lh;

			delete [] pWordList;
			delete [] features;

			printRecFile(recf, res);
		}
		fclose(recf);

	}
	FILE* lhFile = fopen("lhrecord.txt", "w");
	fprintf(lhFile, "lh = %e\n", totalLh);
	fclose(lhFile);

	delete cbset;
	delete dict;
	delete reca;
	delete gbc;
	return 0;
}

