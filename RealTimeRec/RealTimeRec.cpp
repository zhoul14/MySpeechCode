#include "RealTimeRec.h"
#include "../CommonLib/CommonVars.h"
#include "../CommonLib/Feature/20dBEnergyGeometryAverageMFCC45.h"

RealTimeRec::RealTimeRec(const char* codebookFile, const char* dictFile, double durWeight, bool useCuda, bool useSegmentModel, bool useNBest) {
	this->useCuda = useCuda;
	this->useSegmentModel = useSegmentModel;
	this->useNBest = useNBest;
	set = new GMMCodebookSet(codebookFile, GMMCodebookSet::INIT_MODE_FILE);
	gbc = new GMMProbBatchCalc(set, useCuda, useSegmentModel);
	gbc->setDurWeight(durWeight);
	dict = new WordDict(dictFile);

	if (useNBest) {
		nBestRec = new NBestRecAlgorithm();
	} else {
		oneBestRec = new SimpleSpeechRec();
	}

}

std::vector<std::vector<SWord> > RealTimeRec::recSpeech(short* samples, int sampleNum) {
	int frameNum = (sampleNum - FRAME_LEN + FRAME_STEP) / FRAME_STEP;
	auto featureBuf = (float(*)[DIM])malloc(frameNum * DIM * sizeof(float));
	if (featureBuf == NULL ) {
		printf("cannot malloc memory for FeatureBuf\n");
		exit(-1);
	}
	get20dBEnergyGeometryAveragMfcc(samples, featureBuf, frameNum);


	
	float* features = (float*)featureBuf;

	double* fd = new double[frameNum * DIM];
	for (int j = 0; j < frameNum * DIM; j++) {
		fd[j] = features[j];
	}
	gbc->prepare(fd, frameNum);
	delete [] fd;
	free(featureBuf);
	std::vector<std::vector<SWord> > res;
	if (useNBest) {
		res = nBestRec->recSpeech(frameNum, DIM, dict, gbc, -1, useSegmentModel);
		
	} else {
		std::vector<SWord> res0 = oneBestRec->recSpeech(frameNum, DIM, dict, gbc, useSegmentModel,0);
		res.push_back(res0);
	}

	return res;
}

RealTimeRec::~RealTimeRec() {
	if (useNBest) {
		delete nBestRec;
	} else {
		delete oneBestRec;
	}
}