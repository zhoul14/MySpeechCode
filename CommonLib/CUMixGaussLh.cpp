#include "./Math/MathUtility.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "CUMixGaussLh.h"
#include "CommonVars.h"

const double PI = 3.1415926535897;

void CUMixGaussLh::autoSetSepLength() {
	int zMin = 0;
	int yMin = 1;
	double N = fDim;
	cudaDeviceProp prop = getDeviceProp();
	double M = prop.sharedMemPerBlock;
	int Q = prop.maxThreadsPerBlock / 2;

	int zMax = floor(M / N / 256);
	int yMax = floor(M / (4 * N * N + 12 * N + 8));

	int bestProd = 0;
	int bestZ = -1;
	int bestY = -1;
	for (int z = zMin; z <= zMax; z++) {
		for (int y = yMin; y <= yMax; y++) {
			double m = (4 * N * N + 12 * N + 8) * y + 256 * N * z;
			if (y * z > bestProd && m <= M && y * 32 * z <= Q) {
				bestProd = y * z;
				bestZ = z;
				bestY = y;
			}
		}
	}
	cStep = bestY;
	fStep = 32 * bestZ;

}


cudaDeviceProp CUMixGaussLh::getDeviceProp(){
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	return devProp;
}

void CUMixGaussLh::decideCudaFlag(bool cudaFlag) {
	if (cudaFlag) {
		cudaEnabled = false;
		int cudaDeviceCount = -1;
		cudaGetDeviceCount(&cudaDeviceCount);
		for (int i = 0; i < cudaDeviceCount; i++) {
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, i);
			if (deviceProp.major >= 2) {
				cudaEnabled = true;
			}
		}
		if (!cudaEnabled) {
			printf("cannot find qualified cuda device, cuda is not used\n");
		}
	} else {
		cudaEnabled = false;
	}
}


void CUMixGaussLh::init(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag, double *beta, int betanum) {
	this->cbNum = cbNum;
	this->fDim = fDim;
	this->mixNum = -1;

	decideCudaFlag(cudaFlag);


	if (cudaEnabled) {
		this->alphaDev = NULL;
		cudaError_t err;
		if (alpha != NULL) {

			err = cudaMalloc((void**)&alphaDev, cbNum * sizeof(double));
			err = cudaMemcpy(alphaDev, alpha, cbNum * sizeof(double), cudaMemcpyHostToDevice);

			this->mixNum = mixNum;
		}

		autoSetSepLength();

		double* invSigmaCompress = (double*)malloc(cbNum * fDim * (fDim + 1) / 2 * sizeof(double));
		for (int i = 0; i < cbNum; i++) {
			double* invSigmaPtr = invSigma + fDim * fDim * i;
			double* invSigmaCompressPtr = invSigmaCompress + fDim * (fDim + 1) / 2 * i;
			for (int j = 0; j < fDim; j++) {
				for (int k = 0; k < j; k++) {
					invSigmaCompressPtr[j * (j + 1) / 2 + k] = invSigmaPtr[j * fDim + k];
					//*invSigmaCompressPtr = k == j ? 0.5 * t : t;
					//invSigmaCompressPtr++;
				}
				invSigmaCompressPtr[j * (j + 1) / 2 + j] = invSigmaPtr[j * fDim + j] / 2;
			}
		}

		cudaMalloc((void**)&invSigmaDev, cbNum * fDim * (fDim + 1) / 2 * sizeof(double));

		cudaMemcpy(invSigmaDev, invSigmaCompress, cbNum * fDim * (fDim + 1) / 2 * sizeof(double), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&muDev, cbNum * fDim * sizeof(double));
		cudaMemcpy(muDev, mu, cbNum * fDim * sizeof(double), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&invSigmaMuDev, cbNum * fDim * sizeof(double));

		//batchInvSigmaTimesMu();

		cudaMalloc((void**)&cstDev, cbNum * sizeof(double));
		cudaMemcpy(cstDev, cst, cbNum * sizeof(double), cudaMemcpyHostToDevice);
		err = cudaPeekAtLastError();
		//batchInvSigmaMuTimesMu();

		free(invSigmaCompress);
	} 
	else {
		alphaHost = NULL;
		if (alpha != NULL) {
			alphaHost = new double[cbNum];
			memcpy(alphaHost, alpha, cbNum * sizeof(double));
			this->mixNum = mixNum;
		}



		double* invSigmaCompress = new double[cbNum * invSigmaLen];
		
		for (int i = 0; i < cbNum; i++) {
			double* invSigmaPtr1 = invSigma + invSigmaL * i;
			double* invSigmaPtr2 = invSigmaPtr1 + FEATURE_DIM * FEATURE_DIM;
			double* invSigmaCompressPtr1 = invSigmaCompress + invSigmaLen * i;
			double* invSigmaCompressPtr2 = invSigmaCompressPtr1 + invSigmaLen1;
			for (int j = 0; j < FEATURE_DIM; j++) {
				for (int k = 0; k < j; k++) {
					invSigmaCompressPtr1[j * (j + 1) / 2 + k] = invSigmaPtr1[j * FEATURE_DIM + k];
				}
				invSigmaCompressPtr1[j * (j + 1) / 2 + j] = invSigmaPtr1[j * FEATURE_DIM + j] / 2;
			}
			for (int j = 0; j < apFdim; j++)
			{
				for (int k = 0; k < j; k++)
				{
					invSigmaCompressPtr2[j * (j + 1) / 2 + k] = invSigmaPtr2[j * apFdim + k];
				}
				invSigmaCompressPtr2[j * (j + 1) / 2 + j] = invSigmaPtr2[j * apFdim + j] / 2;
			}
		}

		invSigmaHost = new double[cbNum * invSigmaLen];
		memcpy(invSigmaHost, invSigmaCompress, cbNum * invSigmaLen * sizeof(double));

		muHost = new double[cbNum * fDim];
		memcpy(muHost, mu, cbNum * fDim * sizeof(double));

		betaHost = new double [cbNum * betanum];
		memcpy(betaHost, beta, cbNum * betanum * sizeof(double));

		invSigmaMuHost = new double[cbNum * fDim];
		for (int k = 0; k < cbNum; k++) {
			double* r = invSigmaMuHost + k * fDim;
			double* v = muHost + k * fDim;
			double* A = invSigmaHost + k * invSigmaLen;
			double* B = A + invSigmaLen1;

			for (int i = 0; i < FEATURE_DIM; i++) {
				double t = 0;
				for (int j = 0; j < i; j++)
					t += A[i * (i + 1) / 2 + j] * v[j];

				for (int j = i + 1; j < FEATURE_DIM; j++) 
					t += A[j * (j + 1) / 2 + i] * v[j];

				t += A[i * (i + 1) / 2 + i] * 2 * v[i];

				r[i] = t;
			}

			for (int i = 0; i < apFdim; i++) {
				double t = 0;
				for (int j = 0; j < i; j++)
					t += B[i * (i + 1) / 2 + j] * v[j + FEATURE_DIM];

				for (int j = i + 1; j < apFdim; j++) 
					t += B[j * (j + 1) / 2 + i] * v[j + FEATURE_DIM];

				t += B[i * (i + 1) / 2 + i] * 2 * v[i + FEATURE_DIM];

				r[i + FEATURE_DIM] = t;
			}
		}

		cstHost = new double[cbNum * betanum];
		memcpy(cstHost, cst, cbNum * sizeof(double) * betanum);

		for (int k = 0; k < cbNum; k++) {
			double t = 0;
			double* v1 = invSigmaMuHost + k * fDim;
			double* v2 = muHost + k * fDim;
			for (int i = 0; i < FEATURE_DIM; i++) {
				t += v1[i] * v2[i];			
			}
			cstHost[k * 2] += t / 2;
			t = 0;
			for (int i = FEATURE_DIM; i < fDim; i++) {
				t += v1[i] * v2[i];			
			}
			cstHost[k * 2 + 1 ] += t / 2;
		}

		delete [] invSigmaCompress;
	}


}

CUMixGaussLh::CUMixGaussLh(int cbNum, int fDim, double* invSigma, double* mu, bool cudaFlag, double *beta, int betanum){

	apFdim = fDim - FEATURE_DIM; 
	invSigmaLen2 = apFdim * (apFdim + 1) / 2;
	invSigmaLen1 = FEATURE_DIM * (1 + FEATURE_DIM) / 2;
	invSigmaLen = invSigmaLen1 + invSigmaLen2;
	invSigmaL = apFdim * apFdim + FEATURE_DIM * FEATURE_DIM; 

	double c0 = FEATURE_DIM / 2.0 * log(2 * PI);
	double c1 = (fDim - FEATURE_DIM) / 2.0 * log(2 * PI);;
	double *cst = (double*)malloc(cbNum * sizeof(double) * betanum);
	for (int i = 0; i < cbNum * betanum; i ++) {
		double* invSigmaPtr1 = invSigma + i * invSigmaLen;
		double* invSigmaPtr2 = invSigmaPtr1 + invSigmaLen1;
		cst[i * 2] = c0 - 0.5 * log(calcDet(invSigmaPtr1, FEATURE_DIM));
		cst[i * 2 + 1] = c1 - 0.5 * log(calcDet(invSigmaPtr2, apFdim));

	}

	init(cbNum, fDim, invSigma, mu, cst, NULL, 0, cudaFlag, beta, betanum);

	free(cst);
}

CUMixGaussLh::CUMixGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* alpha, int mixNum, bool cudaFlag, double *beta, int betanum){

	apFdim = fDim - FEATURE_DIM; 
	invSigmaLen2 = apFdim * (apFdim + 1) / 2;
	invSigmaLen1 = FEATURE_DIM * (1 + FEATURE_DIM) / 2;
	invSigmaLen = invSigmaLen1 + invSigmaLen2;
	invSigmaL = apFdim * apFdim + FEATURE_DIM * FEATURE_DIM; 

	double c0 = FEATURE_DIM / 2.0 * log(2 * PI);
	double c1 = (fDim - FEATURE_DIM) / 2.0 * log(2 * PI);;
	double *cst = (double*)malloc(cbNum *betanum  * sizeof(double) );
	for (int i = 0; i < cbNum; i ++) {
		double* invSigmaPtr1 = invSigma + i * invSigmaL;
		double* invSigmaPtr2 = invSigmaPtr1 + invSigmaL - apFdim * apFdim;
		cst[i * 2] = c0 - 0.5 * log(calcDet(invSigmaPtr1, FEATURE_DIM));
		cst[i * 2 + 1] = c1 - 0.5 * log(calcDet(invSigmaPtr2, apFdim));
	}

	init(cbNum, fDim, invSigma, mu, cst, alpha, mixNum, cudaFlag, beta, betanum);

	free(cst);
}

CUMixGaussLh::CUMixGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag, double *beta, int betanum){
	apFdim = fDim - FEATURE_DIM; 
	invSigmaLen2 = apFdim * (apFdim + 1) / 2;
	invSigmaLen1 = FEATURE_DIM * (1 + FEATURE_DIM) / 2;
	invSigmaLen = invSigmaLen1 + invSigmaLen2;
	invSigmaL = apFdim * apFdim + FEATURE_DIM * FEATURE_DIM; 
	init(cbNum, fDim, invSigma, mu, cst, alpha, mixNum, cudaFlag, beta, betanum);
}

void CUMixGaussLh::runWeightedCalcInC(double* feature, int fNum, double* res){
	if (alphaHost == NULL) {
		return;
	}
	double* sepLh = new double[mixNum * 2];

	for (int i = 0; i < fNum; i++) {
		for (int j = 0; j < cbNum / mixNum; j++) {
			for (int k = 0; k < mixNum; k++) {
				int c = j * mixNum + k;
				double res1 = cstHost[c * 2];
				double res2 = cstHost[c * 2 + 1];
				int idx = 0;
				for (int l = 0; l < FEATURE_DIM; l++)
				{
					double t1 = feature[i * fDim + l];
					double rest = 0;
					for (int m = 0; m <= l; m++) {
						double t2 = feature[i * fDim + m];
						rest += invSigmaHost[c * invSigmaLen + idx++] * t2;
					}
					rest -= invSigmaMuHost[c * fDim + l];
					res1 += rest * t1;
				}
				for (int l = FEATURE_DIM; l < fDim; l++)
				{
					double t1 = feature[i * fDim + l];
					double rest = 0;
					for (int m = FEATURE_DIM; m <= l; m++) {
						double t2 = feature[i * fDim + m];
						rest += invSigmaHost[c * invSigmaLen + idx++] * t2;
					}
					rest -= invSigmaMuHost[c * fDim + l];

					res2 += rest * t1;

				}
				sepLh[k * 2] = -res1;
				sepLh[k * 2 + 1] = -res2;
			}
			double *alphaPtr = alphaHost + j * mixNum;
			double *betaPtr = betaHost + j * mixNum * 2;
			double weightedRes = MathUtility::logSumExp(sepLh, betaPtr, mixNum * 2);
			res[i * cbNum / mixNum +j] = weightedRes;
		}
	}
	delete [] sepLh;
}

void CUMixGaussLh::runCalcInC(double* feature, int fNum, double* res) {
	for (int i = 0; i < fNum; i++) {
		for (int c = 0; c < cbNum; c++) {
			double reslh1 = cstHost[c * 2];
			double reslh2 = cstHost[c * 2 + 1];
			int idx = 0;
			for (int j = 0; j < FEATURE_DIM; j++) {
				double t1 = feature[i * fDim + j];
				double rest = 0;
				for (int k = 0; k <= j; k++) {
					double t2 = feature[i * fDim + k];
					rest += invSigmaHost[c * invSigmaLen + idx++] * t2;
				}
				rest -= invSigmaMuHost[c * fDim + j];

				reslh1 += rest + t1;
			}
			for (int j = FEATURE_DIM; j < fDim; j++)
			{
				double t1 = feature[i * fDim + FEATURE_DIM + j];
				double rest = 0;
				for (int k = FEATURE_DIM; k <= j; k++) {
					double t2 = feature[i * fDim + FEATURE_DIM + k];
					rest += invSigmaHost[c * invSigmaLen + idx++] * t2;
				}
				rest -= invSigmaMuHost[c * fDim + FEATURE_DIM + j];
				reslh2 = rest * t1;
			}

			int resIdx = c + i * cbNum;
			res[resIdx] = log(exp(-1*(reslh1)) * betaHost[c * 2] + exp(-1*reslh2)*betaHost[c * 2 + 1]);
		}
	}

}

void CUMixGaussLh::runCalcInCuda(double* feature, int fNum, double* res){

}

void CUMixGaussLh::runWeightedCalcInCuda(double* feature, int fNum, double* res){

}

void CUMixGaussLh::runCalc(double* feature, int fNum, double* res) {
	if (cudaEnabled) 
		runCalcInCuda(feature, fNum, res);
	else 
		runCalcInC(feature, fNum, res);
}

void CUMixGaussLh::runWeightedCalc(double* feature, int fNum, double* res) {
	if (cudaEnabled)
		runWeightedCalcInCuda(feature, fNum, res);
	else
		runWeightedCalcInC(feature, fNum, res);
}

CUMixGaussLh::~CUMixGaussLh(){

	if (cudaEnabled) {
		cudaFree(invSigmaDev);
		cudaFree(invSigmaMuDev);
		cudaFree(cstDev);
		cudaFree(muDev);
		cudaFree(alphaDev);
	} else {
		delete [] invSigmaHost;
		delete [] invSigmaMuHost;
		delete [] cstHost;
		delete [] muHost;
		delete [] alphaHost;
		if (betaHost)
		{
			delete []betaHost;
		}
	}

}


void CUMixGaussLh::runCalcInCBeta(double* feature, int fNum, double* res) {

	for (int i = 0; i < fNum; i++) {
		for (int c = 0; c < cbNum ; c++) {
			double reslh1 = cstHost[c * 2];
			double reslh2 = cstHost[c * 2 + 1];
			int idx = 0;
			for (int j = 0; j < FEATURE_DIM; j++) {
				double t1 = feature[i * fDim + j];
				double rest = 0;
				for (int k = 0; k <= j; k++) {
					double t2 = feature[i * fDim + k];
					rest += invSigmaHost[c * invSigmaLen + idx++] * t2;
				}
				rest -= invSigmaMuHost[c * fDim + j];

				reslh1 += rest + t1;
			}
			for (int j = FEATURE_DIM; j < fDim; j++)
			{
				double t1 = feature[i * fDim  + j];
				double rest = 0;
				for (int k = FEATURE_DIM; k <= j; k++) {
					double t2 = feature[i * fDim + k];
					rest += invSigmaHost[c * invSigmaLen + idx++] * t2;
				}
				rest -= invSigmaMuHost[c * fDim + j];
				reslh2 += rest * t1;
			}

			int resIdx = c + i * cbNum * 2;
			res[resIdx] = -reslh1;
			res[resIdx + 1] = -reslh2;
		}
	}

}