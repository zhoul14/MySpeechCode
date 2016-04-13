#ifndef WJQ_CU_GAUSS_LH_H
#define WJQ_CU_GAUSS_LH_H


#include "../Math/CMatrixInv.h"
#include <string.h>
#include "cuda_runtime.h"

class CUGaussLh {
private:
	bool cudaEnabled;

	double m_dCoef;

	int cbNum;

	int fDim;

	int mixNum;

	double *invSigmaDev, *muDev, *invSigmaMuDev, *cstDev, *alphaDev;

	double *invSigmaHost, *muHost, *invSigmaMuHost, *cstHost, *alphaHost;

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

	void init(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag);
	
	void batchInvSigmaTimesMu();

	void batchInvSigmaMuTimesMu();

	double gausslh(double* x, double* mean, double* invsigma, double cst);

	/*void calcLog(char* filename);*/
	void runCalcInCuda(double* feature, int fNum, double* res);

	void runWeightedCalcInCuda(double* feature, int fNum, double* res);

	void runCalcInC(double* feature, int fNum, double* res);

	void runWeightedCalcInC(double* feature, int fNum, double* res);

public:

	CUGaussLh(int cbNum, int fDim, double* invSigma, double* mu, bool cudaFlag, double Coef = 1);

	CUGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* alpha, int mixNum, bool cudaFlag, double Coef = 1);

	CUGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag, double Coef = 1);
	
	~CUGaussLh();

	void runCalc(double* feature, int fNum, double* res);

	void runWeightedCalc(double* feature, int fNum, double* res);
};
#endif