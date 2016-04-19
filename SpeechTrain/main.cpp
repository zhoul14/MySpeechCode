#include "../SpeechSegmentAlgorithm/SegmentAlgorithm.h"
#include "../CommonLib/ReadConfig/TrainParam.h"
#include "../CommonLib/Dict/WordDict.h"
#include "../CommonLib/FileFormat/FeatureFileSet.h"
#include "../SpeechSegmentAlgorithm/SimpleSTFactory.h"
#include "../GMMCodebook/GMMCodebookSet.h"
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../GMMCodebook/GMMUpdateManager.h"
#include <windows.h>
#include <time.h>
#include <cassert>
#include <vector>
#include <string>
#include <cuda.h>
#include "../StateProbabilityMap/StateProbabilityMap.h"
#include "assert.h"
#include "../NBestRecAlgorithm/NBestRecAlgorithm.h"
#include "omp.h"
//#include "vld.h"
using std::vector;
using std::string;

#define PRIME 0
#define STATEPROBMAP 0
#define PRIME_DIM 45
#define WORD_LEVEL_MMIE 1


void summaryUpdateRes(const vector<int>& r, FILE* fid, int iterNum) {
	if (r.size() == 0)
		return;

	vector<int> suc, sne, anz, ic;
	for (int i = 0; i < r.size(); i++) {
		if (r[i] == GMMEstimator::SUCCESS) {
			suc.push_back(i);
		} else if (r[i] == GMMEstimator::SAMPLE_NOT_ENOUGH) {
			sne.push_back(i);
		} else if (r[i] == GMMEstimator::ALPHA_NEAR_ZERO) {
			anz.push_back(i);
		} else if (r[i] == GMMEstimator::ILL_CONDITIONED) {
			ic.push_back(i);
		}
	}
	fprintf(fid, "Train Iter %d\n", iterNum);
	fprintf(fid, "SUCCESS: %d\n", suc.size());
	fprintf(fid, "ALPHA_NEAR_ZERO: %d\n", anz.size());
	for (int i = 0; i < anz.size(); i++) {
		fprintf(fid, "%d ", anz[i]);
	}
	fprintf(fid, "\n");

	fprintf(fid, "SAMPLE_NOT_ENOUGH: %d\n", sne.size());
	for (int i = 0; i < sne.size(); i++) {
		fprintf(fid, "%d ", sne[i]);
	}
	fprintf(fid, "\n");

	fprintf(fid, "ILL_CONDITION: %d\n", ic.size());
	for (int i = 0; i < ic.size(); i++) {
		fprintf(fid, "%d ", ic[i]);
	}
	fprintf(fid, "\n");


	fprintf(fid, "TOTAL: %d\n", r.size());

}

int main(int argc, char** argv) {

	if (argc > 3) {
		printf("usage: program_name config_file [basedir]");
		exit(-1);
	}
	if (argc == 3) {
		if (GetFileAttributes(argv[2]) == INVALID_FILE_ATTRIBUTES) {
			CreateDirectory(argv[2], NULL);
		}
		SetCurrentDirectory(argv[2]);
		//current_path(argv[2]);
	}


	string* configName,* configName2;
	if (argc == 1) {
		configName = new string("train_config.xml");
	} else {
		configName = new string(argv[1]);

#if PRIME
		configName2 = new string(argv[2]);//
#endif

		//configName3 = new string(argv[3]);
	}


	TrainParam tparam(configName->c_str());

#if PRIME
	TrainParam tparam2(configName2->c_str());
#endif 

	bool triPhone = tparam.getTriPhone();
	//////////////////////////////////////////////////////////////////////////
	//for (int crossfitIter = 0; crossfitIter < 50 ;crossfitIter++){
	//////////////////////////////////////////////////////////////////////////	
	//初始化u中各成员
	SegmentUtility u;
	SegmentUtility uHelper;

	SimpleSTFactory* stfact = new SimpleSTFactory();
	SimpleSTFactory* stfactHelper = NULL;

	u.factory = stfact;
	uHelper.factory = stfactHelper;

	WordDict* dict = new WordDict(tparam.getWordDictFileName().c_str(),triPhone);
	u.dict = dict;
	uHelper.dict = dict;
	//dict->setTriPhone(triPhone);
	dict->makeUsingCVidWordList();
	string initCb = tparam.getInitCodebook();

	double Coef = tparam.getDCoef();

	GMMCodebookSet* set = new GMMCodebookSet(initCb.c_str(),0,Coef);
	NBestRecAlgorithm* reca = new NBestRecAlgorithm();

	//set->saveCodebookDSP("preLarge_Model");
#if PRIME
	GMMCodebookSet* mySet = new GMMCodebookSet(857,PRIME_DIM,1,1);//
#endif
	//mySet->mergeIsoCbs2BetaCbs(set);
	//set->printCodebookSetToFile("primeCBSet.txt");
	//mySet->printCodebookSetToFile("MixCBSet.txt");
	/*GMMCodebookSet* testSet= new GMMCodebookSet(857,48,1,3,0);
	testSet->printCodebookSetToFile("testCodebooks.txt");
	*/
	GCSTrimmer::fixSmallDurSigma(set, tparam.getMinDurSigma());


	int splitAddN = tparam.getSplitAddN();
	double splitOffset = tparam.getSplitOffset();
	if (splitAddN > 0) {
		printf("splitting codebooks, add %d mixtures\n", splitAddN);
		if (splitAddN > set->MixNum) {
			printf("splitAddN[%d] param cannot be larger than MixNum[%d]\n", splitAddN, set->MixNum);
			exit(-1);
		}
		set->splitAddN(splitAddN, splitOffset);
	}

	int cbNum = set->getCodebookNum();

	double durWeight = tparam.getDurWeight();
	bool useCuda = tparam.getUseCudaFlag();
	bool useSegmentModel = tparam.getSegmentModelFlag();
	if (useCuda) {
		printf("UseCUDA is true\n");
	} else {
		printf("UseCUDA is false\n");
	}

	if (useSegmentModel) {
		printf("UseSegmentModel is true\n");
	} else {
		printf("UseSegmentModel is false\n");
	}
	GMMProbBatchCalc* gbc =	new GMMProbBatchCalc(set, useCuda, useSegmentModel, Coef);
	gbc->setDurWeight(durWeight);

	u.bc = gbc;
	//初始化输入数据集

	vector<TSpeechFile> inputs = tparam.getTrainFiles();
#if PRIME
	vector<TSpeechFile> inputs2 = tparam2.getTrainFiles();//
#endif 
	SegmentAlgorithm sa;

	int maxEMIter = tparam.getEMIterNum();

	int trainIter = tparam.getTrainIterNum();

	string logPath = tparam.getLogPath();

	string lhRecordPath = logPath + "/lh_record.txt";
	string updateTimePath = logPath + "/update_time.txt";
	string updateIterPath = logPath + "/update_iter.txt";
	string summaryPath = logPath + "/summary.txt";

	FILE* lhRecordFile = fopen(lhRecordPath.c_str(), "w");
	if (!lhRecordFile) {
		printf("cannot open log file[%s]\n", lhRecordPath.c_str());
		exit(-1);
	}

	FILE* updateTimeFile = fopen(updateTimePath.c_str(), "w");
	if (!updateTimeFile) {
		printf("cannot open log file[%s]\n", updateTimePath.c_str());
		exit(-1);
	}

	FILE* summaryFile = fopen(summaryPath.c_str(), "w");
	if (!summaryFile) {
		printf("cannot open log file[%s]\n", summaryPath.c_str());
		exit(-1);
	}
#if PRIME
	GMMUpdateManager ua(mySet, maxEMIter, dict, tparam.getMinDurSigma(), updateIterPath.c_str(), useCuda, useSegmentModel);//
#else
	GMMUpdateManager ua(set, maxEMIter, dict, tparam.getMinDurSigma(), updateIterPath.c_str(), useCuda, useSegmentModel, STATEPROBMAP, 25);
#endif
	/*ua.getMMIEmatrix("ConMatrix.txt",false);
	ua.getMMIEmatrix("DataMatrix.txt",true);*/

	bool* trainedCb = new bool[cbNum];

	double lhOfLastIter = -1e300; 

	for (int iter = 0; iter < trainIter; iter++) {

		clock_t begTime = clock();
		clock_t labTime = 0;
		clock_t prepareTime = 0;
		double lhOfThisIter = 0;

		int trainCnt = -1;

#if STATEPROBMAP
		CStateProbMap CSPM(cbNum);
#endif

		/************************************************start segment*************************************************/
		for (auto i = inputs.begin(); i != inputs.end(); i++) {
			trainCnt++;
			if (trainCnt == tparam.getTrainNum())
				break;
			const int fDim = tparam.getFdim();
			FeatureFileSet input((*i).getFeatureFileName(), (*i).getMaskFileName(), (*i).getAnswerFileName(), fDim);

#if PRIME
			int ii=trainCnt;
			FeatureFileSet input2(inputs2[ii].getFeatureFileName(),inputs2[ii].getMaskFileName(),inputs2[ii].getAnswerFileName(),tparam2.getFdim());
			const int fDim2 = tparam2.getFdim();//
#endif
			int speechNumInFile = input.getSpeechNum();
			for (int j = 0; j < (speechNumInFile); j++) {

				printf("process file %d, speech %d    \r", trainCnt, j);

				int fNum = input.getFrameNumInSpeech(j);

				//if(fNum!=input2.getFrameNumInSpeech(j))//
				//printf("fNum1:%d does not match fNum2:%d",fNum,input2.getFrameNumInSpeech(j));

				int ansNum = input.getWordNumInSpeech(j);

				if (fNum < ansNum * HMM_STATE_DUR * 2) {
					printf("\ntoo short speech, file = %d, speech = %d ignored in training (fNum = %d, ansNum = %d)\n", trainCnt, j, fNum, ansNum);
					continue;
				}

				double* frames;
				frames = new double[fNum * fDim];
				input.getSpeechAt(j, frames);


#if PRIME
				double* frames2 = new double[fDim2 * fNum];
				input2.getSpeechAt(j, frames2);//
#endif
				int* ansList = new int[ansNum];
				input.getWordListInSpeech(j, ansList);
				//分割前完成概率的预计算



				bool* mask = new bool[dict->getTotalCbNum()];
#if !STATEPROBMAP&&!WORD_LEVEL_MMIE
				dict->getUsedStateIdInAns(mask, ansList, ansNum);
				gbc->setMask(mask);
#endif
				clock_t t1 = clock();
				gbc->prepare(frames, fNum);
				clock_t t2 = clock();
				prepareTime += t2 - t1;
				//vector<vector<SWord> > res0 = reca->recSpeech(fNum, fDim, dict, gbc, 4, useSegmentModel);//isoword
				vector<SegmentResult>res0(TOTAL_WORD_NUM);
				t1 = clock();
				SegmentResult res;
				int answer = ansList[0];
				int* usedFrames = NULL;
				usedFrames = new int[fNum * cbNum];
				//for (int c = 0; c != fNum *cbNum; c++)usedFrames[c] = -1;
				memset(usedFrames, 1, sizeof(int) * fNum * cbNum);
				if (WORD_LEVEL_MMIE)
				{	
					omp_set_dynamic(true);
#pragma omp parallel for 
					for (int segIdx = 0; segIdx < TOTAL_WORD_NUM; segIdx++)
					{
						printf("ID: %d, Max threads: %d, Num threads: %d , num procss: %d\n",omp_get_thread_num(), omp_get_max_threads(), omp_get_num_threads(),omp_get_num_procs());  
						ansList[0] = segIdx;
						SegmentResult res1;
						res1 = sa.segmentSpeech(fNum, fDim, ansNum, ansList, u);
						if (segIdx == answer)
						{
							res = res1;
						}
						res0[segIdx] = (res1);					
					}

					for (int segIdx = 0; segIdx != TOTAL_WORD_NUM; segIdx++){
						int totalFrameNum = ua.collect(res0[segIdx].frameLabel, frames, usedFrames, res0[segIdx].lh);
					}
				}
				//input.SaveSegmentPointToBuf(j,res.frameLabel);
				t2 = clock();
				labTime += t2 - t1;

#if PRIME
				int totalFrameNum = ua.collect(res.frameLabel, frames2);//
#else
				int totalFrameNum = ua.collect(res.frameLabel, frames, false);
				ua.collectWordGamma(res0,res.frameLabel,usedFrames);
				//if(!ua.collectWordGamma(res.frameLabel,res0,ansList[0],res.lh))printf("shit !ans is not in recognition result List!\n\n");
				//ua.collectWordGamma(res0, ansList[0], usedFrames);
				delete []usedFrames;

#endif
#if STATEPROBMAP
				CSPM.pushToMap(gbc,res.frameLabel);
#endif

				assert(stfact->getAllocatedNum() == 0);

				lhOfThisIter += res.lh;

#if PRIME
				delete [] frames2;
#endif
				delete [] ansList;
				delete [] frames;
				delete [] mask;
			}
			//input.PrintSegmentPointBuf("SegMent48_log.txt");
			printf("\n");
		}
		/*******************************segment end****************************************/

		clock_t midTime = clock();
		int segTime = (midTime - begTime) / CLOCKS_PER_SEC;
		fprintf(lhRecordFile, "iter %d,\tlh = %e, segment time = %ds(%ds, %ds)", iter, lhOfThisIter, segTime, labTime / CLOCKS_PER_SEC, prepareTime / CLOCKS_PER_SEC);
		fflush(lhRecordFile);

#if STATEPROBMAP
		std::vector<std::vector<double>> m;
		m.resize(cbNum);
		CSPM.mergeMapToMatrix(m, ua.getFW());
		CSPM.saveOutMaptoFile(tparam.getInitCodebook() + "Map.txt", ua.getFW());

		int CorrelateCnt = ua.setMMIEmatrix(m);

		fprintf(lhRecordFile, ", Correlate num = %d", CorrelateCnt);
		fflush(lhRecordFile);
		/*return 0;*/
#endif
		vector<int> updateRes;
		if(STATEPROBMAP)
		{
			for (int uptime = 0; uptime != maxEMIter; uptime++)
			{
				ua.setGBC(gbc);
				updateRes = ua.updateStateLvMMIE();
				set->saveCodebook(tparam.getOutputCodebook());
				summaryUpdateRes(updateRes, summaryFile, iter);
			}
		}
		else if(WORD_LEVEL_MMIE)
		{
			updateRes = ua.updateWordLvMMIE();
		}
		else
		{
			updateRes = ua.update();
		}
		clock_t endTime = clock();

		int updTime = (endTime - midTime) / CLOCKS_PER_SEC;
		fprintf(lhRecordFile, ", update time = %ds\n", updTime);
		fflush(lhRecordFile);


		lhOfLastIter = lhOfThisIter;

		string allCbPath = logPath + "/all_codebooks.txt";
#if PRIME
		mySet->saveCodebook(tparam2.getOutputCodebook());//
		mySet->printCodebookSetToFile(allCbPath.c_str());//
#else
		set->saveCodebook(tparam.getOutputCodebook());
		set->printCodebookSetToFile(allCbPath.c_str());
#endif
	}

	for (int i = 0; i < ua.getUaCbnum(); i++) {
		fprintf(updateTimeFile, "%d\t%d\n", i, ua.getSuccessUpdateTime(i));
	}

	fclose(lhRecordFile);
	fclose(updateTimeFile);
	fclose(summaryFile);
	delete [] trainedCb;	
	delete set;
	delete stfact;
	delete dict;
	delete gbc;	
	delete configName;
	return 0;
}