#ifndef GMMCODEBOOKSET_H
#define GMMCODEBOOKSET_H

#define CUDA_ACCELERATE

#include <string> 
#include "../commonlib/CommonVars.h"
const int g_DefaultFdim = 45;

struct GMMCodebook {

	int FDim, MixNum;
	double*	Mu;//[FEATURE_DIM * MIXTURE_NUM];
	double*	InvSigma;//[FEATURE_DIM * FEATURE_DIM * MIXTURE_NUM];
	double* Alpha; 
	double	DurMean;
	double	DurVar;
	double  DETSIGMA;
	int cbType;
	double* Beta;
	int BetaNum;

	static const int CB_TYPE_DIAG = 0;

	static const int CB_TYPE_FULL_RANK = 1;

	static const int CB_TYPE_FULL_RANK_SHARE = 2;

	static const int CB_TYPE_FULL_MIX = 3;

	int SigmaL() const {
		if (cbType == CB_TYPE_DIAG)
			return MixNum * FDim;

		if (cbType == CB_TYPE_FULL_RANK)
			return MixNum * FDim * FDim;

		if (cbType == CB_TYPE_FULL_RANK_SHARE)
			return FDim * FDim;

		if (cbType == CB_TYPE_FULL_MIX)
			return (FEATURE_DIM * FEATURE_DIM + (FDim - FEATURE_DIM) * (FDim - FEATURE_DIM)) * MixNum;

		printf("error: unrecognized cbType [%d]\n", cbType);
		exit(-1);

		return -1;
	}

	

	GMMCodebook(int mixnum, int fdim, int cbType)
	{
		this->cbType = cbType;
		FDim = fdim;
		MixNum = mixnum;
		int L = SigmaL();
		Mu = new double[fdim * mixnum];
		InvSigma = new double[L];
		Alpha = new double[mixnum];
		DurMean = DurVar = 0;
		BetaNum = 0;
		Beta = NULL;
		if (cbType == CB_TYPE_FULL_MIX)
		{
			BetaNum = 2;
			Beta = new double [MixNum * BetaNum];
		}
		memset(Mu,0,fdim * mixnum);
		memset(InvSigma,1,L);
		memset(Alpha,1/mixnum,mixnum);
	}

	GMMCodebook(const GMMCodebook& g)
	{
		FDim = g.FDim;
		MixNum = g.MixNum;
		//isFullRank = g.isFullRank;
		cbType = g.cbType;
		int L = SigmaL();
		Mu = new double[FDim * MixNum];
		InvSigma = new double[L];
		Alpha = new double[MixNum];

		DurMean = g.DurMean;
		DurVar = g.DurVar;
		DETSIGMA = g.DETSIGMA;
		memcpy(Mu, g.Mu, MixNum * FDim * sizeof(double));
		memcpy(InvSigma, g.InvSigma, L * sizeof(double));
		memcpy(Alpha, g.Alpha, MixNum * sizeof(double));
		if (g.Beta != NULL )
		{
			BetaNum = g.BetaNum;
			Beta = new double[MixNum * BetaNum];
			memcpy(Beta, g.Beta , g.BetaNum * sizeof(double));
		}
		else
		{
			BetaNum = 0;
			Beta = NULL;
		}

}

	GMMCodebook& operator=(const GMMCodebook& rh)
	{
		int L = SigmaL();
		if (MixNum != rh.MixNum || FDim != rh.FDim || cbType != rh.cbType)
		{
			MixNum = rh.MixNum;
			FDim = rh.FDim;
			
			delete [] Mu;
			delete [] InvSigma;
			delete [] Alpha;
			
			Mu = new double[FDim * MixNum];
			InvSigma = new double[L];
			Alpha = new double[MixNum];
		}
		DurMean = rh.DurMean;
		DurVar = rh.DurVar;
		memcpy(Mu, rh.Mu, MixNum * FDim * sizeof(double));
		memcpy(InvSigma, rh.InvSigma, L * sizeof(double));
		memcpy(Alpha, rh.Alpha, MixNum * sizeof(double));

		/*if (rh.Beta != NULL )
		{
			BetaNum = rh.BetaNum;
			Beta = new double[MixNum * BetaNum];
			memcpy(Beta, rh.Beta , rh.BetaNum*sizeof(double));
		}
		else
		{
			BetaNum = 0;
			Beta = NULL;
		}*/
		return *this;
	}

	~GMMCodebook()
	{
		delete [] Mu;
		delete [] InvSigma;
		delete [] Alpha;
		if (Beta!=NULL)
		{
			delete [] Beta;
		}
	}
};



class GMMCodebookSet
{
	friend class GCSTrimmer;

	friend class GMMProbBatchCalc;

//public:
private:
	bool initializeFromOutside;

	

	//int*	updateTime;

	void renewDetSigma(int* Idx, int updateNum);

	void updateCodebook(int* Idx, int updateNum, double* alpha, double* mu, double* invsigma, double* durmean, double* durvar, double* beta =NULL);
	
	int SigmaL() const {
		if (cbType == CB_TYPE_DIAG)
			return MixNum * FDim;

		if (cbType == CB_TYPE_FULL_RANK)
			return MixNum * FDim * FDim;

		if (cbType == CB_TYPE_FULL_RANK_SHARE)
			return FDim * FDim;

		if (cbType == CB_TYPE_FULL_MIX)
			return (FEATURE_DIM * FEATURE_DIM + (FDim - FEATURE_DIM) * (FDim - FEATURE_DIM)) * MixNum;

		printf("error: unrecognized cbType [%d]\n", cbType);
		exit(-1);

		return -1;
	}

	int SigmaL(int mixNum) const {
		if (cbType == CB_TYPE_DIAG)
			return mixNum * FDim;

		if (cbType == CB_TYPE_FULL_RANK)
			return mixNum * FDim * FDim;

		if (cbType == CB_TYPE_FULL_RANK_SHARE)
			return FDim * FDim;

		if (cbType == CB_TYPE_FULL_MIX)
			return (FEATURE_DIM * FEATURE_DIM + (FDim - FEATURE_DIM) * (FDim - FEATURE_DIM)) * mixNum;

		printf("error: unrecognized cbType [%d]\n", cbType);
		exit(-1);

		return -1;
	}

	bool isSharedSigma() const {
		return cbType == CB_TYPE_FULL_RANK_SHARE;
	} 

	void splitTwoMu(double* invSigmaPtr, double* oldMuPtr, double* newMuPtr1, double* newMuPtr2, double offset);

	void fillZeroAlpha(double offset);

	void myBubbleSort(double* val, int n, int* idx);

public:

	int CodebookNum;

	int FDim;

	int MixNum;

	int cbType;

	double* Alpha;

	double* Mu;

	double* InvSigma;

	double m_Coef;

	double* DurMean;

	double* DurVar;	//durvar=dursigma^2

	double* LogDurVar;

	double* DetSigma;	//det(Sigma)

	int BetaNum;

	double * Beta;

	static const int INIT_MODE_FILE = 0;

	static const int INIT_MODE_MEMORY = 1;

	static const int CB_TYPE_DIAG = 0;

	static const int CB_TYPE_FULL_RANK = 1;

	static const int CB_TYPE_FULL_RANK_SHARE = 2;

	static const int CB_TYPE_FULL_MIX = 3;

	GMMCodebookSet(int cbnum, int fdim, int mixnum, int cbType, double coef = 0.0f);

	GMMCodebookSet(const char* initFile, int mode = INIT_MODE_FILE, double coef = 0.0f);

	~GMMCodebookSet();

	//void reduceMixNum(int n);

	//void reduceOneMix();

	//void deleteOneMix(int* dlist);

	int getCodebookNum() const;

	int getFDim() const;

	int getMixNum() const;

	bool isFullRank() const;

	bool isFullMix() const;

	bool saveCodebook(const std::string& filename);

	int saveCodebookDSP(char * FileName);

	void printCodebookSetToFile(const char* filename);

	//void printUninterruptedUpdateTimes(const char* filename);

	GMMCodebook getCodebook(int num);

	int getCbType() const;

	void updateCodebook(int cbIdx, GMMCodebook& cb);

	void split2(double offset);

	void splitAddN(int addN, double offset);

	void checkEqual(GMMCodebookSet* other);

	bool mergeIsoCbs2BetaCbs(GMMCodebookSet* IsoCbset);

	void writeCBset2File(const std::string& filename);
};

class GCSTrimmer {
public:
	static void fixSmallDurSigma(GMMCodebookSet* set, double minSigma) {
		int n = set->CodebookNum;
		double* durVar = set->DurVar;
		double minVar = minSigma * minSigma;
		for (int i = 0; i < n; i++) {
			if (durVar[i] < minVar) {
				durVar[i] = minVar;
			}
		}
	}
};




#endif