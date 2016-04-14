#ifndef WJQ_GMM_UPDATE_MANAGER_H
#define WJQ_GMM_UPDATE_MANAGER_H

#include "MMIEstimator.h"
#include "GMMEstimator.h"
#include "../CommonLib/Dict/WordDict.h"
#include "GMMCodebookSet.h"
#include "FrameWarehouse.h"
#include "string"

class GMMUpdateManager {
private:

	GMMCodebookSet* codebooks;

	CMMIEstimator* estimator;

	double* firstMoment;

	double* secondMoment;

	double* firstMomentOfLog;

	int* successUpdateTime;

	int updateIter;

	std::vector<std::vector<double>>SimilarMat;

	int* durCnt;

	bool useSegmentModel;

	bool m_bMMIE;

	WordDict* dict;

	double **m_pMMIEres;

	double **m_pMMIEgen;

	double minDurVar;

	FrameWarehouse* fw;

	GMMProbBatchCalc* gbc;

	FILE* logFile;

	std::vector<std::vector<int>> m_ConMatrix;//different data same state

	std::vector<std::vector<int>> m_DataMatrix;//different state same data

	std::vector<std::vector<double>> m_WordGamma;//different WordGamma

	double m_dDsm;

	int makeMMIEmatrix();

	void prepareMMIELh();

public:

	GMMUpdateManager(GMMCodebookSet* codebooks, int maxIter, WordDict* dict, double minDurSigma, const char* logFileName, bool useCuda, bool useSegmentModel, bool m_bMMIE, double Dsm);

	//void setShareCovFlag (int flag);

	//返回值表示执行完本次函数后累积的全部码本的总帧数
	int collect(const std::vector<int>& frameLabel, double* frames);

	int collectWordGamma(const std::vector<int>& frameLabel, double* frames, const std::vector<double>& recLh, int ans);

	int getUaCbnum(){return codebooks->CodebookNum;}

	//返回值为长度为cbnum的vector，vector的每个元素代表相应的码本的更新结果
	std::vector<int> update();

	FrameWarehouse* getFW(){return fw;}

	int getSuccessUpdateTime(int cbidx) const;

	void setGBC(GMMProbBatchCalc* pGbc){
		gbc = pGbc;
	}
	int makeConvMatrix();

	std::vector<std::vector<double>> &getSimilarMat(){
		return SimilarMat;
	}

	bool getMMIEmatrix(std::string filename, bool bDataState);

	int setMMIEmatrix(std::vector<std::vector<double>>&x);

	void setSimilarMatrix(std::vector<std::vector<double>>&x){
		SimilarMat = x;
	}

	void mergeIntoVec(vector<int> vec, vector<vector<int>>&v, bool* bList);

	void updateSMLmat(int idx);

	~GMMUpdateManager();
};

#endif