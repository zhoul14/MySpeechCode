#ifndef ZL_MMIESTIMATOR_H
#define  ZL_MMIESTIMATOR_H


#include "GMMEstimator.h"

class CMMIEstimator :public GMMEstimator
{
public:

    int estimate();

	CMMIEstimator(int fDim, int mixNum, int cbType, int maxIter, bool useCuda, double coef, int betaNum);

	void initMMIE(int ConMatLen, double m_dDsm);

	void initMMIECbs(int DataMatLen);

	inline void setSelfIdx(int idx){
		m_nSelfProc = idx;
	}

	void printfGamma(FILE* logFile = stderr);

	void destoyMMIEr();

	void destroyMMIECbs();

	inline void getDataMat(std::vector<std::vector<int>>& m){
		DataStateMat = m;
	}

	~CMMIEstimator();

	bool destructMat(void* pMat, int len);

	void loadMMIEData(double* d, int dataNum, int idx, int dataId);

	

	void loadMMIECbs(double* a, double* m, double * invSgm, int idx);

	double prepareGammaIForUpdate(int idx, bool bFirst);

	bool prepareByMMIELh(int CBidx, int fnum);

	void setGamma1(int k);
	
	bool checkGamma(double&res);

	void setMMIELh(double** p, double** pGen){
		m_vMMIELH = p;
		m_vMMIELHgen = pGen;
	}


private:
    
	int MMIElen;

	int MMIEDataLen;

	double m_dDsm;

	std::vector<int> m_vDataNum;

	std::vector<int> m_vDataId;

	int* m_pDataMatSelfIdx;

	double** m_vNormc;

	double** m_vDataList;

	double** m_vPrecalcLh;

	double* m_pMMIEgamma;

	double** m_vTgamma;

	double** m_vMuList;

	double** m_vInvList;

	double** m_vAlphaList;

	double** m_vMMIELH;

	double** m_vMMIELHgen;

	int m_nSelfProc;
	
	bool singleGaussianMMIEupdate();

	std::vector<std::vector<int>> DataStateMat;

	//UpdateInfo updateOnce();

	bool MixtureGaussianMMIEupdate();

};

#endif // !1
