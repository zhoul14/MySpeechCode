#ifndef	WJQ_TRAIN_PERAMETER_H
#define	WJQ_TRAIN_PERAMETER_H

#include <string>
#include <vector>

class TSpeechFile {
	friend class TrainParam;

private:
	std::string featureFile;

	std::string maskFile;

	std::string answerFile;

public:


	std::string getFeatureFileName() {
		return featureFile;
	}

	std::string getMaskFileName() {
		return maskFile;
	}

	std::string getAnswerFileName() {
		return answerFile;
	}
};

class TrainParam {
private:
	static const int SHARE_NONE = 0;
	static const int SHARE_ALL = 1;
	static const int SHARE_NON_DIAG = 2;

	std::string initCodebook;

	std::string wordDictFileName;

	std::string outputCodebook;

	std::string logPath;

	double durWeight;

	int trainNum;

	int splitAddN;

	double splitOffset;

	bool useHelperCb;

	bool triPhone;

	int fDim;

	double Coef;

	std::vector<TSpeechFile> fileSets;

	int useCuda;

	int trainIterNum;

	int EMIterNum;

	int useSegmentModel;

	double minDurSigma;

	void checkExist(const std::string& str);

	void checkExist(const char* str);

	void trim(std::string& s);

public:

	TrainParam(const char* filename);

	std::string getWordDictFileName();

	std::string getInitCodebook();

	std::string getOutputCodebook();

	int getFdim();

	double getDurWeight();

	int getTrainNum();

	bool getTriPhone();

	double getMinDurSigma();

	bool getUseCudaFlag();

	bool getSegmentModelFlag();

	int getEMIterNum();

	int getTrainIterNum();

	int getSplitTime();

	int getSplitAddN();

	double getSplitOffset();

	double getDCoef();

	std::vector<TSpeechFile> getTrainFiles();

	std::string getLogPath();
};

#endif