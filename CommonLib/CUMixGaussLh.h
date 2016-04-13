#ifndef ZL_CU_MIXGAUSS_LH_H
#define ZL_CU_MIXGAUSS_LH_H


#include "./Math/CMatrixInv.h"
#include <string.h>
#include "CommonVars.h"
#include "cuda_runtime.h"

class CUMixGaussLh {
private:
	bool cudaEnabled;

	int cbNum;

	int fDim;

	int mixNum;

	double *invSigmaDev, *muDev, *invSigmaMuDev, *cstDev, *alphaDev, *betaDev;

	double *invSigmaHost, *muHost, *invSigmaMuHost, *cstHost, *alphaHost, *betaHost;

	int apFdim,invSigmaLen2,invSigmaLen1,invSigmaLen,invSigmaL;

	int fStep, cStep;

	double calcDet(double* mat, int fDim) {
		double* t = new double[fDim * fDim];
		memcpy(t, mat, fDim * fDim * sizeof(double));
		double det = MatrixInv(t, fDim);
		delete [] t;
		return det;
	}
	void decideCudaFlag(bool cudaFlag);

	void autoSetSepLength();

	cudaDeviceProp getDeviceProp();

	void init(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag, double *beta = NULL, int betanum = 0);

	//void batchInvSigmaTimesMu();

	//void batchInvSigmaMuTimesMu();

	//double gausslh(double* x, double* mean, double* invsigma, double cst);

	/*void calcLog(char* filename);*/
	void runCalcInCuda(double* feature, int fNum, double* res);

	void runWeightedCalcInCuda(double* feature, int fNum, double* res);

	void runCalcInC(double* feature, int fNum, double* res);

	void runWeightedCalcInC(double* feature, int fNum, double* res);

public:

	CUMixGaussLh(int cbNum, int fDim, double* invSigma, double* mu, bool cudaFlag, double *beta = NULL, int betanum = 0);

	CUMixGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* alpha, int mixNum, bool cudaFlag, double *beta = NULL, int betanum = 0);

	CUMixGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag, double *beta = NULL, int betanum = 0);

	~CUMixGaussLh();

	void runCalc(double* feature, int fNum, double* res);

	void runWeightedCalc(double* feature, int fNum, double* res);

	void runCalcInCBeta(double* feature, int fNum, double* res);
};











#endif