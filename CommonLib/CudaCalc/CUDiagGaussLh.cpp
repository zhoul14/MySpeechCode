#include "CUDiagGaussLh.h"
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <cuda.h>
#include <string.h>
#include "../Math/MathUtility.h"

extern "C" void kernelDiagMvnWrapper(int cbNum, int fDim, double* invSigmaDev, double* muDev, double* cstDev, double* featureDev, int fNum, double* resBufDev, dim3 threads, dim3 blocks, int memSize);

extern "C" void kernelLSEWrapper(int cbNum, int fStep, int mixNum, double* alphaDev, double* seperateLhDev, double* combineLhDev);

void CUDiagGaussLh::decideCudaFlag(bool cudaFlag) {
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

void CUDiagGaussLh::init(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag) {
	this->cbNum = cbNum;
	this->fDim = fDim;
	this->mixNum = -1;
	decideCudaFlag(cudaFlag);

	if (cudaEnabled) {
		this->alphaDev = NULL;
		autoSetSepLength();

		cudaError_t err;
		if (alpha != NULL) {
			err = cudaMalloc((void**)&alphaDev, cbNum * sizeof(double));
			err = cudaMemcpy(alphaDev, alpha, cbNum * sizeof(double), cudaMemcpyHostToDevice);
			this->mixNum = mixNum;
		}

		err = cudaMalloc((void**)&invSigmaDev, cbNum * fDim * sizeof(double));
		err = cudaMemcpy(invSigmaDev, invSigma, cbNum * fDim * sizeof(double), cudaMemcpyHostToDevice);

		err = cudaMalloc((void**)&muDev, cbNum * fDim * sizeof(double));
		err = cudaMemcpy(muDev, mu, cbNum * fDim * sizeof(double), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&cstDev, cbNum * sizeof(double));
		cudaMemcpy(cstDev, cst, cbNum * sizeof(double), cudaMemcpyHostToDevice);
	} else {
		this->alphaHost = NULL;
		//autoSetSepLength();

		//cudaError_t err;
		if (alpha != NULL) {
			alphaHost = new double[cbNum];
			memcpy(alphaHost, alpha, cbNum * sizeof(double));
			//err = cudaMalloc((void**)&alphaDev, cbNum * sizeof(double));
			//err = cudaMemcpy(alphaDev, alpha, cbNum * sizeof(double), cudaMemcpyHostToDevice);
			this->mixNum = mixNum;
		}

		invSigmaHost = new double[cbNum * fDim];
		memcpy(invSigmaHost, invSigma, cbNum * fDim * sizeof(double));

// 		err = cudaMalloc((void**)&invSigmaDev, cbNum * fDim * sizeof(double));
// 		err = cudaMemcpy(invSigmaDev, invSigma, cbNum * fDim * sizeof(double), cudaMemcpyHostToDevice);

		muHost = new double[cbNum * fDim];
		memcpy(muHost, mu, cbNum * fDim * sizeof(double));
// 		err = cudaMalloc((void**)&muDev, cbNum * fDim * sizeof(double));
// 		err = cudaMemcpy(muDev, mu, cbNum * fDim * sizeof(double), cudaMemcpyHostToDevice);

		cstHost = new double[cbNum];
		memcpy(cstHost, cst, cbNum * sizeof(double));

// 		cudaMalloc((void**)&cstDev, cbNum * sizeof(double));
// 		cudaMemcpy(cstDev, cst, cbNum * sizeof(double), cudaMemcpyHostToDevice);
	}

	
}

CUDiagGaussLh::CUDiagGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* alpha, int mixNum, bool cudaFlag) {

	const double PI = 3.1415926535897;
	double c0 = fDim / 2.0 * log(2 * PI);
	double* cst = (double*)malloc(cbNum * sizeof(double));
	for (int i = 0; i < cbNum; i++) {
		double* invSigmaPtr = invSigma + i * fDim;
		double det = 0;
		for (int j = 0; j < fDim; j++)
			det += log(invSigmaPtr[j]);

		cst[i] = c0 - 0.5 * det;
	}
	

	init(cbNum, fDim, invSigma, mu, cst, alpha, mixNum, cudaFlag);

	free(cst);
}

CUDiagGaussLh::CUDiagGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag) {
	init(cbNum, fDim, invSigma, mu, cst, alpha, mixNum, cudaFlag);
}

cudaDeviceProp CUDiagGaussLh::getDeviceProp() {
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	return devProp;
}

//y:一次处理的码本个数（即cStep）
//z:32*z是一次处理的帧数（即32*z=fStep),使用32的原因是32是warp的长度
//码本长度:(16*N+8)*y
//帧长度8*N*x (x = 32*z)
//所以满足(16*N+8)*y+8*N*x<=M(shared mem 总长度）
void CUDiagGaussLh::autoSetSepLength() {
	int zMin = 0;
	int yMin = 1;
	double N = fDim;
	cudaDeviceProp prop = getDeviceProp();
	double M = prop.sharedMemPerBlock;
	int Q = prop.maxThreadsPerBlock / 2;

	int zMax = floor(M / N / 256);
	int yMax = floor(M / (16 * N + 8));

	int bestProd = 0;
	int bestZ = -1;
	int bestY = -1;
	for (int z = zMin; z <= zMax; z++) {
		for (int y = yMin; y <= yMax; y++) {
			double m = (16 * N + 8) * y + 256 * N * z;
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

void CUDiagGaussLh::runCalcInC(double* feature, int fNum, double* res) {
	for (int i = 0; i < fNum; i++) {
		for (int c = 0; c < cbNum; c++) {
			double reslh = 0;
			for (int j = 0; j < fDim; j++) {
				double xj = feature[i * fDim + j] - muHost[c * fDim + j];
				reslh += xj * xj * invSigmaHost[c * fDim + j];
			}
			reslh = -(cstHost[c] + reslh / 2);
			res[c + i * cbNum] = reslh;
		}
	}
}

void CUDiagGaussLh::runCalcInCuda(double* feature, int fNum, double* res) {

	cudaError_t err;

	double* featureDev;
	int fExtendNum = ((fNum + fStep - 1) / fStep) * fStep;
	err = cudaMalloc((void**)&featureDev, fStep * fDim * sizeof(double));
	err = cudaMemset(featureDev, 0, fStep * fDim * sizeof(double));


	double* resDev;
	err = cudaMalloc((void**)&resDev, fStep * cbNum * sizeof(double));
	cudaMemset(resDev, 0, fStep * cbNum * sizeof(double));

	int memSize = (2 * fDim + 1) * cStep * sizeof(double) + fStep * fDim * sizeof(double);
	int M = getDeviceProp().sharedMemPerBlock;
	if (memSize > M) {
		printf("memsize error in cuDiagGaussLh\n");
		exit(-1);
	}

	dim3 threads(fStep, cStep);
	dim3 blocks((cbNum + cStep - 1) / cStep, 1);

	for (int i = 0; i < fExtendNum / fStep; i++) {

		int fs = fStep;
		if (i == fExtendNum / fStep - 1) {
			fs = fNum - i * fStep;
		}
		err = cudaMemcpy(featureDev, feature + i * fStep * fDim, fs * fDim * sizeof(double), cudaMemcpyHostToDevice);
		kernelDiagMvnWrapper(cbNum, fDim, invSigmaDev, muDev, cstDev, featureDev, fStep, resDev, threads, blocks, memSize);	
		err = cudaMemcpy(res + i * cbNum * fStep, resDev, fs * cbNum * sizeof(double), cudaMemcpyDeviceToHost);

	}

	err = cudaFree(resDev);
	err = cudaFree(featureDev);

}

void CUDiagGaussLh::runWeightedCalcInC(double* feature, int fNum, double* res) {

	double* sepLh = new double[mixNum];
	for (int i = 0; i < fNum; i++) {
		for (int j = 0; j < cbNum / mixNum; j++) {
			for (int k = 0; k < mixNum; k++) {
				int c = j * mixNum + k;
				double reslh = 0;
				for (int l = 0; l < fDim; l++) {
					double xt = feature[i * fDim + l] - muHost[c * fDim + l];
					reslh += xt * xt * invSigmaHost[c * fDim + l];
				}
				reslh = -(cstHost[c] + reslh / 2);
				sepLh[k] = reslh;
			}
			res[i * cbNum / mixNum + j] = MathUtility::logSumExp(sepLh, alphaHost + j * mixNum, mixNum);
		}
	}
	delete [] sepLh;
}

void CUDiagGaussLh::runWeightedCalcInCuda(double* feature, int fNum, double* res) {
	if (alphaDev == NULL) {
		return;
	}
	cudaError_t err;

	double* featureDev;
	int fExtendNum = ((fNum + fStep - 1) / fStep) * fStep;
	err = cudaMalloc((void**)&featureDev, fStep * fDim * sizeof(double));
	err = cudaMemset(featureDev, 0, fStep * fDim * sizeof(double));


	double* resDev;
	err = cudaMalloc((void**)&resDev, fStep * cbNum * sizeof(double));
	cudaMemset(resDev, 0, fStep * cbNum * sizeof(double));

	double* weightedResDev;
	err = cudaMalloc((void**)&weightedResDev, fStep * (cbNum / mixNum) * sizeof(double));
	cudaMemset(weightedResDev, 0, fStep * (cbNum / mixNum) * sizeof(double));

	int memSize = (2 * fDim + 1) * cStep * sizeof(double) + fStep * fDim * sizeof(double);
	int M = getDeviceProp().sharedMemPerBlock;
	if (memSize > M) {
		printf("memsize error in cuDiagGaussLh\n");
		exit(-1);
	}

	dim3 threads(fStep, cStep);
	dim3 blocks((cbNum + cStep - 1) / cStep, 1);
	
	for (int i = 0; i < fExtendNum / fStep; i++) {

		int fs = fStep;
		if (i == fExtendNum / fStep - 1) {
			fs = fNum - i * fStep;
		}

		err = cudaMemcpy(featureDev, feature + i * fStep * fDim, fs * fDim * sizeof(double), cudaMemcpyHostToDevice);
		kernelDiagMvnWrapper(cbNum, fDim, invSigmaDev, muDev, cstDev, featureDev, fStep, resDev, threads, blocks, memSize);	
		kernelLSEWrapper(cbNum / mixNum, fs, mixNum, alphaDev, resDev, weightedResDev);
		err = cudaMemcpy(res + i * (cbNum / mixNum) * fStep, weightedResDev, fs * (cbNum / mixNum) * sizeof(double), cudaMemcpyDeviceToHost);

	}

	err = cudaFree(resDev);
	err = cudaFree(featureDev);
	err = cudaFree(weightedResDev);

}


CUDiagGaussLh::~CUDiagGaussLh() {
	if (cudaEnabled) {
		cudaFree(invSigmaDev);
		cudaFree(muDev);
		cudaFree(cstDev);
		cudaFree(alphaDev);
	} else {
		delete [] invSigmaHost;
		delete [] muHost;
		delete [] cstHost;
		delete [] alphaHost;
	}

}

void CUDiagGaussLh::runCalc(double* feature, int fNum, double* res) {
	if (cudaEnabled)
		runCalcInCuda(feature, fNum, res);
	else 
		runCalcInC(feature, fNum, res);
}

void CUDiagGaussLh::runWeightedCalc(double* feature, int fNum, double* res) {
	if (cudaEnabled)
		runWeightedCalcInCuda(feature, fNum, res);
	else
		runWeightedCalcInC(feature, fNum, res);
}