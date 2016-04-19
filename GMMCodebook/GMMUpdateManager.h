#ifndef WJQ_GMM_UPDATE_MANAGER_H
#define WJQ_GMM_UPDATE_MANAGER_H

#include "MMIEstimator.h"
#include "GMMEstimator.h"
#include "../CommonLib/Dict/WordDict.h"
#include "GMMCodebookSet.h"
#include "FrameWarehouse.h"
#include "WordLevelMMIEstimator.h"
#include "string"
#include "../SpeechSegmentAlgorithm/SegmentAlgorithm.h"

class SegmentResult;
class GMMUpdateManager {
private:

	GMMCodebookSet* codebooks;

	GMMEstimator* estimator;

	double* firstMoment;

	double* secondMoment;

	double* firstMomentOfLog;

	int* successUpdateTime;

	int updateIter;

	std::vector<std::vector<double>>SimilarMat;

	int* durCnt;

	bool useSegmentModel;

	bool m_bMMIE;

	bool m_bCuda;

	WordDict* dict;

	double **m_pMMIEres;

	double **m_pMMIEgen;

	double minDurVar;

	double ObjectFuncVal;

	FrameWarehouse* fw;

	GMMProbBatchCalc* gbc;

	FILE* logFile;

	std::vector<std::vector<int>> m_ConMatrix;//different data same state

	std::vector<std::vector<int>> m_DataMatrix;//different state same data

	std::vector<std::vector<double>> m_WordGamma;//different WordGamma

	std::vector<int>m_WordGammaLastPos;

	double m_dDsm;

	int makeMMIEmatrix();

	void prepareMMIELh();

public:

	GMMUpdateManager(GMMCodebookSet* codebooks, int maxIter, WordDict* dict, double minDurSigma, const char* logFileName, bool useCuda, bool useSegmentModel, bool m_bMMIE, double Dsm);

	//void setShareCovFlag (int flag);\

	int getWordGammaTotalNum();

	//����ֵ��ʾִ���걾�κ������ۻ���ȫ���뱾����֡��
	int collect(const std::vector<int>& frameLabel, double* frames, bool bCollectFrame = true);

	double collectWordGamma(const std::vector<SegmentResult>& recLh, const std::vector<int>& frameLabel, int *pushedFrames);

	int collect(const std::vector<int>& frameLabel, double* frames, int *pushedFrames, double& lh);

	int collectWordGamma(const std::vector<int>& frameLabel, std::vector<SWord>& recLh, int ans, double segLh);

	int collectWordGamma(const std::vector<SegmentResult>& recLh, int wordIdx, bool* usedFrame);

	int getUaCbnum(){return codebooks->CodebookNum;}

	//����ֵΪ����Ϊcbnum��vector��vector��ÿ��Ԫ�ش�����Ӧ���뱾�ĸ��½��
	std::vector<int> update();

	std::vector<int> updateStateLvMMIE();

	std::vector<int> updateWordLvMMIE();	

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