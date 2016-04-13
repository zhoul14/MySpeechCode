#ifndef WJQ_GMM_ESTIMATOR_H
#define WJQ_GMM_ESTIMATOR_H

#include <vector>
#include <string>
const double PI = 3.1415926535897;
const unsigned int DefaultFdim = 45;
struct UpdateInfo {
	int status;

	int mixIdx;

	double lhBeforeUpdate;
};


class GMMEstimator {
	protected:

	int fDim;

	int mixNum;

	int cbType;

	double* data;

	double* alpha;

	double* mu;

	double* invSigma;

	double* normc;

	int maxIter;

	int dataNum;

	//int shareCovFlag;

	double* precalcLh;

	double m_dCoef;

	int cbId;

	double* beta;
	
	int betaNum;

	double* precalcBetaLh;

	FILE* logFile;

	virtual UpdateInfo updateOnce();

	UpdateInfo diagUpdate();

	UpdateInfo updateMix();

	//void probOfMixture(int mixIdx, int dataIdx, double* prob, double* lh);

	void updateNormC();

	bool isAlphaTooSmall(double* softTrainNum);

	bool isRcondTooSmall(double* mat);

	void dumpData(char* filename);

	void prepareLhForUpdate();

	void prepareLhForUpdateMIx();

	bool singleGaussianUpdate();

	void dumpData();

	int sigmaL() {
		if (cbType == CB_TYPE_FULL_MIX)
		{
			return DefaultFdim * DefaultFdim + (fDim - DefaultFdim) * (fDim - DefaultFdim);
		}
		return cbType != CB_TYPE_DIAG ? fDim * fDim : fDim;
	}

	void splitMixtureByAlpha(int idx);

	bool shareInvSigma() {
		return cbType == CB_TYPE_FULL_RANK_SHARE;
	}

	bool isFullRank() {
		return cbType != CB_TYPE_DIAG && cbType != CB_TYPE_FULL_MIX;
	}

	void init(int fDim, int mixNum, int cbType, int maxIter, bool useCuda);

	bool useCuda;

public:
	static const int SUCCESS = 0;
	static const int SAMPLE_NOT_ENOUGH = 1;
	static const int ILL_CONDITIONED = 2;
	static const int ALPHA_NEAR_ZERO = 3;

	static const int SHARE_NONE = 0;
	static const int SHARE_ALL = 1;
	static const int SHARE_NON_DIAG = 2;

	static const int CB_TYPE_DIAG = 0;
	static const int CB_TYPE_FULL_RANK = 1;
	static const int CB_TYPE_FULL_RANK_SHARE = 2;
	static const int CB_TYPE_FULL_MIX = 3;

	GMMEstimator(int fDim, int mixNum, int cbType, int maxIter, bool useCuda, double Coef = 1, int BetaNum = 2);

	GMMEstimator(int fDim, int mixNum, int cbType, int maxIter, bool useCuda);
		
	void setCbId(int id);

	virtual	~GMMEstimator();

	void loadParam(double* alpha, double* mu, double* invSigma, double* beta = NULL);

	void loadData(double* x, int dataNum);

	void saveParam(double* alpha, double* mu, double* invSigma, double* beta = NULL);

	void setOutputFile(FILE* fid);

	int estimate();

	void loadMMIEParam();

	int	inline getMaxIter(){
		return maxIter;
	}

	static std::string errorInfo(int e);
};

#endif