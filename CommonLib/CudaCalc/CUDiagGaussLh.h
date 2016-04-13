#ifndef _WJQ_CU_DIAG_GAUSS_LH_H_
#define _WJQ_CU_DIAG_GAUSS_LH_H_
#include "cuda_runtime.h"


class CUDiagGaussLh {

	bool cudaEnabled;

	int cbNum;

	int fDim;

	int mixNum;

	double *invSigmaDev, *muDev, *cstDev, *alphaDev;

	double *invSigmaHost, *muHost, *cstHost, *alphaHost;

	int fStep, cStep;

	cudaDeviceProp getDeviceProp();

	void autoSetSepLength();

	void init(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag);

	void decideCudaFlag(bool cudaFlag);

	void runCalcInCuda(double* feature, int fNum, double* res);

	void runWeightedCalcInCuda(double* feature, int fNum, double* res);

	void runCalcInC(double* feature, int fNum, double* res);

	void runWeightedCalcInC(double* feature, int fNum, double* res);

public:
	CUDiagGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag);

	CUDiagGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* alpha, int mixNum, bool cudaFlag);

	void runCalc(double* feature, int fNum, double* res);

	void runWeightedCalc(double* feature, int fNum, double* res);

	~CUDiagGaussLh();
};


#endif