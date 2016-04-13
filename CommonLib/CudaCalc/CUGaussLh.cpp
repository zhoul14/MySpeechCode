#include "CUGaussLh.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../Math/MathUtility.h"
#include <float.h>
#define DEFAULT_FDIM 45

extern "C" void kernelMvnWrapper(int cbNum, int fDim, double* invSigmaDev, double* muDev, double* cstDev, double* featureDev, int fNum, double* resBufDev, dim3 threads, dim3 blocks, int memSize);

extern "C" void kernelDgemvWrapper(double* invSigmaDev, double* muDev, int fDim, int cbNum, int shareNum, double* resDev , dim3 threads, dim3 blocks, int memSize);

extern "C" void kernelVecDotWrapper(double* invSigmaMuDev, double* muDev, int fDim, int cbNum, double* resDev , dim3 threads, dim3 blocks);

extern "C" void kernelLSEWrapper(int cbNum, int fStep, int mixNum, double* alphaDev, double* seperateLhDev, double* combineLhDev);


cudaDeviceProp CUGaussLh::getDeviceProp() {
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	return devProp;
}


//y:一次处理的码本个数（即cStep）
//z:32*z是一次处理的帧数（即32*z=fStep),使用32的原因是32是warp的长度
//码本长度:(4*N^2+12*N+8)*y
//帧长度8*N*x
//所以满足(4*N^2+12*N+8)*y+8*N*x<=M(shared mem 总长度）
void CUGaussLh::autoSetSepLength() {
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

void CUGaussLh::runCalcInC(double* feature, int fNum, double* res) {
	int invSigmaLen = fDim * (fDim + 1) / 2;
	for (int i = 0; i < fNum; i++) {
		for (int c = 0; c < cbNum; c++) {
			double reslh = cstHost[c];
			int idx = 0;
			for (int j = 0; j < fDim; j++) {
				double t1 = feature[i * fDim + j];
				double rest = 0;
				for (int k = 0; k <= j; k++) {
					double t2 = feature[i * fDim + k];
					rest += invSigmaHost[c * invSigmaLen + idx++] * t2;
				}

				rest -= invSigmaMuHost[c * fDim + j];
				if (j < DEFAULT_FDIM)
				{
					reslh += rest * t1;
				}
				else
				{
					reslh += rest * t1 * m_dCoef;
				}
			}

			int resIdx = c + i * cbNum;
			res[resIdx] = -reslh;
		}
	}

}

// void CUGaussLh::runCalcInC(double* mu, double* invSigma, double* cst, double* feature, int fNum, double* res) {
// 	for (int i = 0; i < fNum; i++) {
// 		for (int j = 0; j < cbNum; j++) {
// 			double* muptr = mu + fDim * j;
// 			double* isptr = invSigma + fDim * fDim * j;
// 			double c = cst[j];
// 			double* x = feature + i * fDim;
// 			res[i * cbNum + j] = gausslh(x, muptr, isptr, c);
// 		}
// 	}
// }
// 
// void CUGaussLh::runCalcInC(double* mu, double* invSigma, double* feature, int fNum, double* res) {
// 	const double PI = 3.1415926535897;
// 	double c0 = fDim / 2.0 * log(2 * PI);
// 	double* cst = (double*)malloc(cbNum * sizeof(double));
// 	for (int i = 0; i < cbNum; i++) {
// 		double* invSigmaPtr = invSigma + i * fDim * fDim;
// 		cst[i] = c0 - 0.5 * log(calcDet(invSigmaPtr, fDim));
// 	}
// 
// 	for (int i = 0; i < fNum; i++) {
// 		for (int j = 0; j < cbNum; j++) {
// 			double* muptr = mu + fDim * j;
// 			double* isptr = invSigma + fDim * fDim * j;
// 			double c = cst[j];
// 			double* x = feature + i * fDim;
// 			res[i * cbNum + j] = gausslh(x, muptr, isptr, c);
// 		}
// 	}
// 	free(cst);
// 
// }

double CUGaussLh::gausslh(double* x, double* mean, double* invsigma, double cst) {

	double res = cst;
	double* t = new double[fDim];
	for (int i = 0; i < fDim; i++)
		t[i] = x[i] - mean[i];

	for (int i = 0; i < fDim; i++) {
		double rest = 0;
		for (int j = 0; j < i; j++) 
			rest += invsigma[i * fDim + j] * t[j];
		rest += invsigma[i * fDim + i] * t[i] / 2;
		res += rest * t[i];
	}

	delete [] t;
	return -res;
}

CUGaussLh::CUGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag, double coef) {
	m_dCoef = coef;
	init(cbNum, fDim, invSigma, mu, cst, alpha, mixNum, cudaFlag);	
}

CUGaussLh::CUGaussLh(int cbNum, int fDim, double* invSigma, double* mu, bool cudaFlag, double coef) {
	const double PI = 3.1415926535897;
	m_dCoef = coef;
	double c0 = fDim / 2.0 * log(2 * PI);
	if (coef<1e-6)
	{
		c0 = DEFAULT_FDIM / 2.0 * log(2 * PI);
	}
	double* cst = (double*)malloc(cbNum * sizeof(double));
	for (int i = 0; i < cbNum; i++) {
		double* invSigmaPtr = invSigma + i * fDim * fDim;
		cst[i] = c0 - 0.5 * log(calcDet(invSigmaPtr, fDim));
	}
	init(cbNum, fDim, invSigma, mu, cst, NULL, 0, cudaFlag);
	free(cst);

}

void CUGaussLh::decideCudaFlag(bool cudaFlag) {
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

void CUGaussLh::init(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* alpha, int mixNum, bool cudaFlag) {
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

		err = cudaMalloc((void**)&invSigmaDev, cbNum * fDim * (fDim + 1) / 2 * sizeof(double));
		if(err != CUDA_SUCCESS){
			printf("invSigmaDev cuda malloc error!\n");
		}

		err = cudaMemcpy(invSigmaDev, invSigmaCompress, cbNum * fDim * (fDim + 1) / 2 * sizeof(double), cudaMemcpyHostToDevice);
		if(err != CUDA_SUCCESS){
			printf("cudaMemcpy invSigmaDev cuda cpy error!\n");
		}

		err = cudaMalloc((void**)&muDev, cbNum * fDim * sizeof(double));
		if(err != CUDA_SUCCESS){
			printf("muDev cuda malloc error!\n");
		}

		err = cudaMemcpy(muDev, mu, cbNum * fDim * sizeof(double), cudaMemcpyHostToDevice);
		if(err != CUDA_SUCCESS){
			printf("cudaMemcpy muDev cuda cpy error!\n");
		}
		err = cudaMalloc((void**)&invSigmaMuDev, cbNum * fDim * sizeof(double));
		if(err != CUDA_SUCCESS){
			printf("invSigmaMuDev cuda malloc error!\n");
		}
		batchInvSigmaTimesMu();

		cudaMalloc((void**)&cstDev, cbNum * sizeof(double));
		cudaMemcpy(cstDev, cst, cbNum * sizeof(double), cudaMemcpyHostToDevice);
		err = cudaPeekAtLastError();
		batchInvSigmaMuTimesMu();

		free(invSigmaCompress);
	} else {
		alphaHost = NULL;
		if (alpha != NULL) {
			alphaHost = new double[cbNum];
			memcpy(alphaHost, alpha, cbNum * sizeof(double));
			this->mixNum = mixNum;
		}


		int invSigmaLen = fDim * (fDim + 1) / 2;
		double* invSigmaCompress = new double[cbNum * invSigmaLen];
		for (int i = 0; i < cbNum; i++) {
			double* invSigmaPtr = invSigma + fDim * fDim * i;
			double* invSigmaCompressPtr = invSigmaCompress + invSigmaLen * i;
			for (int j = 0; j < fDim; j++) {
				for (int k = 0; k < j; k++) {
					invSigmaCompressPtr[j * (j + 1) / 2 + k] = invSigmaPtr[j * fDim + k];
				}
				invSigmaCompressPtr[j * (j + 1) / 2 + j] = invSigmaPtr[j * fDim + j] / 2;
			}
		}

		invSigmaHost = new double[cbNum * invSigmaLen];
		memcpy(invSigmaHost, invSigmaCompress, cbNum * invSigmaLen * sizeof(double));

		muHost = new double[cbNum * fDim];
		memcpy(muHost, mu, cbNum * fDim * sizeof(double));

		invSigmaMuHost = new double[cbNum * fDim];
		for (int k = 0; k < cbNum; k++) {
			double* r = invSigmaMuHost + k * fDim;
			double* v = muHost + k * fDim;
			double* A = invSigmaHost + k * invSigmaLen;
			for (int i = 0; i < fDim; i++) {
				double t = 0;
				for (int j = 0; j < i; j++)
					t += A[i * (i + 1) / 2 + j] * v[j];

				for (int j = i + 1; j < fDim; j++) 
					t += A[j * (j + 1) / 2 + i] * v[j];

				t += A[i * (i + 1) / 2 + i] * 2 * v[i];

				r[i] = t;
			}

		}

		cstHost = new double[cbNum];
		memcpy(cstHost, cst, cbNum * sizeof(double));

		for (int k = 0; k < cbNum; k++) {
			double t = 0;
			double* v1 = invSigmaMuHost + k * fDim;
			double* v2 = muHost + k * fDim;
			for (int i = 0; i < fDim; i++) {
				if(i < DEFAULT_FDIM)
				{
					t += v1[i] * v2[i];
				}
				else
				{
					t += v1[i] * v2[i] * m_dCoef;
				}
			}
			cstHost[k] += t / 2;
		}

		delete [] invSigmaCompress;
	}


}

CUGaussLh::CUGaussLh(int cbNum, int fDim, double* invSigma, double* mu, double* alpha, int mixNum, bool cudaFlag , double coef) {

	const double PI = 3.1415926535897;
	double c0 = fDim / 2.0 * log(2 * PI);
	if (coef<1e-6)
	{
		c0 = DEFAULT_FDIM / 2.0 * log(2 * PI);
	}
	double* cst = (double*)malloc(cbNum * sizeof(double));
	for (int i = 0; i < cbNum; i++) {
		double* invSigmaPtr = invSigma + i * fDim * fDim;
		cst[i] = c0 - 0.5 * log(calcDet(invSigmaPtr, fDim));
	}
	m_dCoef = coef;
	init(cbNum, fDim, invSigma, mu, cst, alpha, mixNum, cudaFlag);
	free(cst);

}

void CUGaussLh::batchInvSigmaMuTimesMu() {
	int bd = 512;
	int gd = (cbNum + bd - 1) / bd;
	dim3 threads(bd, 1);
	dim3 blocks(gd, 1);

	kernelVecDotWrapper(invSigmaMuDev, muDev, fDim, cbNum, cstDev, threads, blocks);
}

void CUGaussLh::batchInvSigmaTimesMu() {

	int M = getDeviceProp().sharedMemPerBlock;
	int bd = (M / fDim / sizeof(double) / 32) * 32;	//128
	int gd = (cbNum + bd - 1) / bd;  

	dim3 threads(bd, 1);
	dim3 blocks(gd, 1);
	int memSize = bd * fDim * sizeof(double);
	if (memSize > M) {
		printf("wrong memsize %d\n", memSize);
		exit(-1);
	}

	int shareNum = 1;
	kernelDgemvWrapper(invSigmaDev, muDev, fDim, cbNum, shareNum, invSigmaMuDev, threads, blocks, memSize);

}

void CUGaussLh::runWeightedCalcInCuda(double* feature, int fNum, double* res) {
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

	int memSize = (fDim * (fDim + 1) / 2 + fDim + 1) * cStep * sizeof(double) + fStep * fDim * sizeof(double);
	int M = getDeviceProp().sharedMemPerBlock;
	if (memSize > M) {
		printf("process %d codebooks and %d frames in one batch, cost %d bytes of shared memory, more than %d bytes", cStep, fStep, memSize, M);
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
		kernelMvnWrapper(cbNum, fDim, invSigmaDev, invSigmaMuDev, cstDev, featureDev, fStep, resDev, threads, blocks, memSize);	
		kernelLSEWrapper(cbNum / mixNum, fs, mixNum, alphaDev, resDev, weightedResDev);
		err = cudaMemcpy(res + i * (cbNum / mixNum) * fStep, weightedResDev, fs * (cbNum / mixNum) * sizeof(double), cudaMemcpyDeviceToHost);

	}
	err = cudaFree(resDev);
	err = cudaFree(featureDev);
	err = cudaFree(weightedResDev);

}

void CUGaussLh::runWeightedCalcInC(double* feature, int fNum, double* res) {
	if (alphaHost == NULL) {
		return;
	}
	int invSigmaLen = fDim * (fDim + 1) / 2;
	double* sepLh = new double[mixNum];
	for (int i = 0; i < fNum; i++) {
		for (int j = 0; j < cbNum / mixNum; j++) {
			for (int k = 0; k < mixNum; k++) {
				int c = j * mixNum + k;
				double res = cstHost[c];
				int idx = 0;
				for (int l = 0; l < fDim; l++) {
					double t1 = feature[i * fDim + l];
					double rest = 0;
					for (int m = 0; m <= l; m++) {
						double t2 = feature[i * fDim + m];
						rest += invSigmaHost[c * invSigmaLen + idx++] * t2;
					}
					rest -= invSigmaMuHost[c * fDim + l];
					if(l < DEFAULT_FDIM)
					{
						res += rest * t1;
					}
					else
					{
						res += rest *t1 * m_dCoef;
					}
				}
				sepLh[k] = -res;
			}
			double* alphaptr = alphaHost + j * mixNum;
			double weightedRes = MathUtility::logSumExp(sepLh, alphaptr, mixNum);

			res[i * cbNum / mixNum + j] = weightedRes;
		}
	}
	delete [] sepLh;

}

void CUGaussLh::runCalcInCuda(double* feature, int fNum, double* res) {
	cudaError_t err;

	double* featureDev;
	int fExtendNum = ((fNum + fStep - 1) / fStep) * fStep;
	err = cudaMalloc((void**)&featureDev, fStep * fDim * sizeof(double));
	err = cudaMemset(featureDev, 0, fStep * fDim * sizeof(double));


	double* resDev;
	err = cudaMalloc((void**)&resDev, fStep * cbNum * sizeof(double));
	cudaMemset(resDev, 0, fStep * cbNum * sizeof(double));

	int memSize = (fDim * (fDim + 1) / 2 + fDim + 1) * cStep * sizeof(double) + fStep * fDim * sizeof(double);
	int M = getDeviceProp().sharedMemPerBlock;
	if (memSize > M) {
		printf("process %d codebooks and %d frames in one batch, cost %d bytes of shared memory, more than %d bytes", cStep, fStep, memSize, M);
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
		kernelMvnWrapper(cbNum, fDim, invSigmaDev, invSigmaMuDev, cstDev, featureDev, fStep, resDev, threads, blocks, memSize);	
		err = cudaMemcpy(res + i * cbNum * fStep, resDev, fs * cbNum * sizeof(double), cudaMemcpyDeviceToHost);
	}


	err = cudaFree(resDev);
	err = cudaFree(featureDev);
}

CUGaussLh::~CUGaussLh() {
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
	}

}

void CUGaussLh::runCalc(double* feature, int fNum, double* res) {
	if (cudaEnabled) 
		runCalcInCuda(feature, fNum, res);
	else 
		runCalcInC(feature, fNum, res);
}

void CUGaussLh::runWeightedCalc(double* feature, int fNum, double* res) {
	if (cudaEnabled)
		runWeightedCalcInCuda(feature, fNum, res);
	else
		runWeightedCalcInC(feature, fNum, res);
}