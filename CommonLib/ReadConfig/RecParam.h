#ifndef	WJQ_REC_PERAMETER_H
#define	WJQ_REC_PERAMETER_H

#include <string>
#include <vector>

class RSpeechFile {
	friend class RecParam;

private:
	std::string featureFile;

	std::string maskFile;

	std::string answerFile;

	std::string recResultFile;

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

	std::string getRecResultFileName() {
		return recResultFile;
	}
};

class RecParam {
private:
	std::string codebookFileName;

	std::string wordDictFileName;

	int recNum;

	int multiJump;

	int useCuda;

	int fDim;

	double Coef;

	bool triPhone;

	std::vector<RSpeechFile> fileSets;

	double durWeight;

	int useSegmentModel;

	bool m_bHeadNOise;

	void checkExist(const char* str);

	void checkExist(const std::string& str);

	void trim(std::string& s);

public:
	RecParam(char* filename);

	std::string getCodebookFileName();

	std::string getWordDictFileName();

	int getRecNum();

	int getMultiJump();

	bool getUseCudaFlag();

	bool getSegmentModelFlag();

	double getDurWeight();

	std::vector<RSpeechFile> getRecFiles();

	bool getTriPhone();

	int getFdim();

	bool getBHeadNoise();

	double getDCoef();
};

#endif