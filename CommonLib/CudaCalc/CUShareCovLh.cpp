#include "CUShareCovLh.h"
#include "../Math/MathUtility.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

extern "C" void kernelLSEWrapper(int cbNum, int fStep, int mixNum, double* alphaDev, double* seperateLhDev, double* combineLhDev);

extern "C" void kernelSharedMvnWrapper(int cbNum, int fDim, int shareNum, double* invSigmaDev, double* featureDev, int fNum, double* resBufDev, dim3 threads, dim3 blocks, int memSize);

extern "C" void kernelShareAddCstWrapper(int cbNum, int fNum, double* cstDev, double* resDev ,dim3 threads, dim3 blocks);

extern "C" void kernelDgemvWrapper(double* invSigmaDev, double* muDev, int fDim, int cbNum, int shareNum, double* resDev, dim3 threads, dim3 blocks, int memSize);

extern "C" void kernelVecDotWrapper(double* invSigmaMuDev, double* muDev, int fDim, int cbNum, double* resDev , dim3 threads, dim3 blocks);

extern "C" void kernelVecDot2Wrapper(double* invSigmaMuDev, double* featureDev, int fDim, int cbNum, int fNum, double* resDev , dim3 threads, dim3 blocks, int memSize);


cudaDeviceProp CUShareCovLh::getDeviceProp() {
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	return devProp;
}


//y:一次处理的码本个数（即cStep）
//z:32*z是一次处理的帧数（即32*z=fStep),使用32的原因是32是warp的长度
//码本长度:(4*N^2+4*N)*y
//帧长度8*N*x
//所以满足(4*N^2+4*N)*y+8*N*x<=M(shared mem 总长度）
void CUShareCovLh::autoSetSepLength() {
	int zMin = 0;
	int yMin = 1;
	double N = fDim;
	cudaDeviceProp prop = getDeviceProp();
	double M = prop.sharedMemPerBlock;
	int Q = prop.maxThreadsPerBlock / 2;

	int zMax = floor(M / N / 256);
	int yMax = floor(M / (4 * N * N + 4 * N));

	int bestProd = 0;
	int bestZ = -1;
	int bestY = -1;
	for (int z = zMin; z <= zMax; z++) {
		for (int y = yMin; y <= yMax; y++) {
			double m = (4 * N * N + 4 * N) * y + 256 * N * z;
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

void CUShareCovLh::decideCudaFlag(bool cudaFlag) {
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


void CUShareCovLh::init(int cbNum, int fDim, int mixNum, double* invSigma, double* mu, double* cst, double* alpha, bool cudaFlag) {

	this->cbNum = cbNum;
	this->fDim = fDim;
	this->mixNum = mixNum;
	decideCudaFlag(cudaFlag);
	if (cudaEnabled) {
		this->alphaDev = NULL;
		cudaError_t err;
		if (alpha != NULL) {
			err = cudaMalloc((void**)&alphaDev, cbNum * sizeof(double));
			err = cudaMemcpy(alphaDev, alpha, cbNum * sizeof(double), cudaMemcpyHostToDevice);
		}

		autoSetSepLength();

		double* invSigmaCompress = (double*)malloc(cbNum / mixNum * fDim * (fDim + 1) / 2 * sizeof(double));
		for (int i = 0; i < cbNum / mixNum; i++) {
			double* invSigmaPtr = invSigma + fDim * fDim * i;
			double* invSigmaCompressPtr = invSigmaCompress + fDim * (fDim + 1) / 2 * i;
			for (int j = 0; j < fDim; j++) {
				for (int k = 0; k < j; k++) {
					invSigmaCompressPtr[j * (j + 1) / 2 + k] = invSigmaPtr[j * fDim + k];
				}
				invSigmaCompressPtr[j * (j + 1) / 2 + j] = invSigmaPtr[j * fDim + j] / 2;
			}
		}

		cudaMalloc((void**)&invSigmaDev, cbNum / mixNum * fDim * (fDim + 1) / 2 * sizeof(double));
		cudaMemcpy(invSigmaDev, invSigmaCompress, cbNum / mixNum * fDim * (fDim + 1) / 2 * sizeof(double), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&muDev, cbNum * fDim * sizeof(double));
		cudaMemcpy(muDev, mu, cbNum * fDim * sizeof(double), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&invSigmaMuDev, cbNum * fDim * sizeof(double));
		batchInvSigmaTimesMu();

		cudaMalloc((void**)&cstDev, cbNum * sizeof(double));
		double* cstTmp = new double[cbNum];
		for (int i = 0; i < cbNum / mixNum; i++) {
			for (int j = 0; j < mixNum; j++) {
				cstTmp[i * mixNum + j] = cst[i];
			}
		}
		cudaMemcpy(cstDev, cstTmp, cbNum * sizeof(double), cudaMemcpyHostToDevice);
		delete [] cstTmp;

		batchInvSigmaMuTimesMu();

		free(invSigmaCompress);
	} else {
		this->alphaHost = NULL;
		if (alpha != NULL) {
			alphaHost = new double[cbNum];
			memcpy(alphaHost, alpha, cbNum * sizeof(double));
			//err = cudaMalloc((void**)&alphaDev, cbNum * sizeof(double));
			//err = cudaMemcpy(alphaDev, alpha, cbNum * sizeof(double), cudaMemcpyHostToDevice);
		}

		//autoSetSepLength();
		int invSigmaLen = fDim * (fDim + 1) / 2;
		double* invSigmaCompress = new double[cbNum / mixNum * invSigmaLen];
		//double* invSigmaCompress = (double*)malloc(cbNum / mixNum * fDim * (fDim + 1) / 2 * sizeof(double));
		for (int i = 0; i < cbNum / mixNum; i++) {
			double* invSigmaPtr = invSigma + fDim * fDim * i;
			double* invSigmaCompressPtr = invSigmaCompress + invSigmaLen * i;
			for (int j = 0; j < fDim; j++) {
				for (int k = 0; k < j; k++) {
					invSigmaCompressPtr[j * (j + 1) / 2 + k] = invSigmaPtr[j * fDim + k];
				}
				invSigmaCompressPtr[j * (j + 1) / 2 + j] = invSigmaPtr[j * fDim + j] / 2;
			}
		}

		invSigmaHost = new double[cbNum / mixNum * invSigmaLen];
		memcpy(invSigmaHost, invSigmaCompress, cbNum / mixNum * invSigmaLen * sizeof(double));

		//cudaMalloc((void**)&invSigmaDev, cbNum / mixNum * invSigmaLen * sizeof(double));
		//cudaMemcpy(invSigmaDev, invSigmaCompress, cbNum / mixNum * invSigmaLen * sizeof(double), cudaMemcpyHostToDevice);

		muHost = new double[cbNum * fDim];
		memcpy(muHost, mu, cbNum * fDim * sizeof(double));

		//cudaMalloc((void**)&muDev, cbNum * fDim * sizeof(double));
		//cudaMemcpy(muDev, mu, cbNum * fDim * sizeof(double), cudaMemcpyHostToDevice);

		invSigmaMuHost = new double[cbNum * fDim];
		for (int c = 0; c < cbNum; c++) {
			double* r = invSigmaMuHost + c * fDim;
			double* v = muHost + c * fDim;
			double* A = invSigmaHost + (c / mixNum) * invSigmaLen;
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
		//cudaMalloc((void**)&invSigmaMuDev, cbNum * fDim * sizeof(double));
		//batchInvSigmaTimesMu();

		cstHost = new double[cbNum];
		//cudaMalloc((void**)&cstDev, cbNum * sizeof(double));
		double* cstTmp = new double[cbNum];
		for (int i = 0; i < cbNum / mixNum; i++) {
			for (int j = 0; j < mixNum; j++) {
				cstTmp[i * mixNum + j] = cst[i];
			}
		}
		memcpy(cstHost, cstTmp, cbNum * sizeof(double));
		//cudaMemcpy(cstDev, cstTmp, cbNum * sizeof(double), cudaMemcpyHostToDevice);
		delete [] cstTmp;


		for (int k = 0; k < cbNum; k++) {
			double t = 0;
			double* v1 = invSigmaMuHost + k * fDim;
			double* v2 = muHost + k * fDim;
			for (int i = 0; i < fDim; i++) {
				t += v1[i] * v2[i];
			}
			cstHost[k] += t / 2;
		}
		//batchInvSigmaMuTimesMu();

		delete [] invSigmaCompress;
	}

	

}

void CUShareCovLh::batchInvSigmaTimesMu() {

	int M = getDeviceProp().sharedMemPerBlock;
	int bd = (M / fDim / sizeof(double) / 32) * 32;	//128
	int gd = (cbNum + bd - 1) / bd;  

	dim3 threads(bd, 1);
	dim3 blocks(gd, 1);
	int memSize = bd * fDim * sizeof(double);

	kernelDgemvWrapper(invSigmaDev, muDev, fDim, cbNum, mixNum, invSigmaMuDev, threads, blocks, memSize);

}

void CUShareCovLh::batchInvSigmaMuTimesMu() {
	int bd = 512;
	int gd = (cbNum + bd - 1) / bd;
	dim3 threads(bd, 1);
	dim3 blocks(gd, 1);

	kernelVecDotWrapper(invSigmaMuDev, muDev, fDim, cbNum, cstDev, threads, blocks);
}


CUShareCovLh::CUShareCovLh(int cbNum, int fDim, int shareNum, double* invSigma, double* mu, double* alpha, bool cudaFlag) {

	const double PI = 3.1415926535897;
	double c0 = fDim / 2.0 * log(2 * PI);
	double* cst = (double*)malloc(cbNum / shareNum * sizeof(double));
	for (int i = 0; i < cbNum / shareNum; i++) {
		double* invSigmaPtr = invSigma + i * fDim * fDim;
		cst[i] = c0 - 0.5 * log(calcDet(invSigmaPtr, fDim));
	}
	init(cbNum, fDim, shareNum, invSigma, mu, cst, alpha, cudaFlag);
	free(cst);

}

CUShareCovLh::CUShareCovLh(int cbNum, int fDim, int shareNum, double* invSigma, double* mu, double* cst, double* alpha, bool cudaFlag) {
	init(cbNum, fDim, shareNum, invSigma, mu, cst, alpha, cudaFlag);	
}

CUShareCovLh::~CUShareCovLh() {
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

// void CUShareCovLh::runCalcInC(double* mu, double* invSigma, double* feature, int fNum, double* res) {
// 	const double PI = 3.1415926535897;
// 	double c0 = fDim / 2.0 * log(2 * PI);
// 	double* cst = new double[cbNum / mixNum];
// 	for (int i = 0; i < cbNum / mixNum; i++) {
// 		double* invSigmaPtr = invSigma + i * fDim * fDim;
// 		cst[i] = c0 - 0.5 * log(calcDet(invSigmaPtr, fDim));
// 	}
// 
// 	for (int i = 0; i < fNum; i++) {
// 		printf("processing frame %d/%d\n", i, fNum);
// 		for (int j = 0; j < cbNum; j++) {
// 			double* muptr = mu + fDim * j;
// 			int sigmaIdx = j / mixNum;
// 			double* isptr = invSigma + fDim * fDim * sigmaIdx;
// 			double c = cst[sigmaIdx];
// 			double* x = feature + i * fDim;
// 			res[i * cbNum + j] = gausslh(x, muptr, isptr, c);
// 		}
// 	}
// 
// 
// 	delete [] cst;
// }

// void CUShareCovLh::runWeightedCalcInC(double* feature, int fNum, double* invSigma, double* mu, double* alpha, double* res) {
// 	const double PI = 3.1415926535897;
// 	double c0 = fDim / 2.0 * log(2 * PI);
// 	double* cst = new double[cbNum / mixNum];
// 	for (int i = 0; i < cbNum / mixNum; i++) {
// 		double* invSigmaPtr = invSigma + i * fDim * fDim;
// 		cst[i] = c0 - 0.5 * log(calcDet(invSigmaPtr, fDim));
// 	}
// 	double* tmplh = new double[mixNum];
// 	for (int i = 0; i < fNum; i++) {
// 		printf("processing frame %d/%d\n", i, fNum);
// 		double* x = feature + i * fDim;
// 		for (int j = 0; j < cbNum / mixNum; j++) {
// 			double* isptr = invSigma + fDim * fDim * j;
// 			double* alphaptr = alpha + mixNum * j;
// 			double c = cst[j];
// 			for (int k = 0; k < mixNum; k++) {
// 				double* muptr = mu + fDim * (j * mixNum + k);
// 				tmplh[k] = gausslh(x, muptr, isptr, c);
// 			}
// 			res[i * cbNum / mixNum + j] = MathUtility::logSumExp(tmplh, alphaptr, mixNum);
// 		}
// 	}
// 	delete [] tmplh;
// 
// 
// 
// }

double CUShareCovLh::gausslh(double* x, double* mean, double* invsigma, double cst) {

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


void CUShareCovLh::runCalcInC(double* feature, int fNum, double* res) {
	int invSigmaLen = (fDim + 1) * fDim / 2;
	memset(res, 0, fNum * cbNum * sizeof(double));
	for (int i = 0; i < fNum; i++) {
		for (int c = 0; c < cbNum / mixNum; c++) {
			double resval = 0;
			int idx = 0;
			for (int j = 0; j < fDim; j++) {
				double t1 = feature[i * fDim + j];
				double rest = 0;
				for (int k = 0; k <= j; k++) {
					double t2 = feature[i * fDim + k];
					rest += invSigmaHost[c * invSigmaLen + idx++] * t2;
				}
				//rest += InvSigma[threadIdx.y * invSigmaLen + idx++] * t1;
				resval += rest * t1;
			}

			int resIdxStart = (c + i * (cbNum / mixNum)) * mixNum;
			for (int j = 0; j < mixNum; j++) {
				res[resIdxStart + j] -= resval;
			}
		}

		for (int c = 0; c < cbNum; c++) {
			int resIdx = i * cbNum + c;
			double* v1 = feature + i * fDim;
			double* v2 = invSigmaMuHost + c * fDim;
			double resval = 0;
			for (int j = 0; j < fDim; j++) {
				resval += v1[j] * v2[j];
			}
			res[resIdx] += resval;
		}

		double* fVec = res + i * cbNum;
		for (int j = 0; j < cbNum; j++)
			fVec[j] -= cstHost[j];
	}

}

void CUShareCovLh::runWeightedCalcInC(double* feature, int fNum, double* res) {
	int invSigmaLen = (fDim + 1) * fDim / 2;
	double* sepLh = new double[mixNum];
	memset(res, 0, fNum * (cbNum / mixNum) * sizeof(double));
	for (int i = 0; i < fNum; i++) {
		for (int c = 0; c < cbNum / mixNum; c++) {
			double resval1 = 0;
			int idx = 0;
			for (int j = 0; j < fDim; j++) {
				double t1 = feature[i * fDim + j];
				double rest = 0;
				for (int k = 0; k <= j; k++) {
					double t2 = feature[i * fDim + k];
					rest += invSigmaHost[c * invSigmaLen + idx++] * t2;
				}
				resval1 += rest * t1;
			}
			
			for (int m = 0; m < mixNum; m++) {
				int p = c * mixNum + m;
				double* v1 = feature + i * fDim;
				double* v2 = invSigmaMuHost + p * fDim;
				double resval2 = 0;
				for (int j = 0; j < fDim; j++) {
					resval2 += v1[j] * v2[j];
				}
				sepLh[m] = -resval1 + resval2 - cstHost[p];
			}
			double* alphaPtr = alphaHost + c * mixNum;
			res[i * (cbNum / mixNum) + c] = MathUtility::logSumExp(sepLh, alphaPtr, mixNum);
		}
	}
}

void CUShareCovLh::runCalcInCuda(double* feature, int fNum, double* res) {
	cudaError_t err;

	double* featureDev;
	int fExtendNum = ((fNum + fStep - 1) / fStep) * fStep;
	err = cudaMalloc((void**)&featureDev, fStep * fDim * sizeof(double));
	err = cudaMemset(featureDev, 0, fStep * fDim * sizeof(double));


	double* resDev;
	err = cudaMalloc((void**)&resDev, fStep * cbNum * sizeof(double));
	cudaMemset(resDev, 0, fStep * cbNum * sizeof(double));

	int memSize = fDim * (fDim + 1) / 2 * cStep * sizeof(double) + fStep * fDim * sizeof(double);
	int M = getDeviceProp().sharedMemPerBlock;
	if (memSize > M) {
		printf("process %d codebooks and %d frames in one batch, cost %d bytes of shared memory, more than %d bytes", cStep, fStep, memSize, M);
		exit(-1);
	}

	dim3 threads(fStep, cStep);
	dim3 blocks((cbNum + cStep - 1) / cStep, 1);

	int cStep2 = (M / sizeof(double) - fStep * fDim) / fDim;
	if (cStep2 > 1024 / fStep)
		cStep2 = 1024 / fStep;
	dim3 threads2(fStep, cStep2);
	dim3 blocks2((cbNum + cStep2 - 1) / cStep2, 1);
	int memSize2 = (cStep2 + fStep) * fDim * sizeof(double);

	dim3 threads3(512, 1);
	dim3 blocks3(fStep, 1);

	for (int i = 0; i < fExtendNum / fStep; i++) {

		int fs = fStep;
		if (i == fExtendNum / fStep - 1) {
			fs = fNum - i * fStep;
		}

		err = cudaMemcpy(featureDev, feature + i * fStep * fDim, fs * fDim * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemset(resDev, 0, fStep * cbNum * sizeof(double));
		kernelSharedMvnWrapper(cbNum, fDim, mixNum, invSigmaDev, featureDev, fStep, resDev, threads, blocks, memSize);
		kernelVecDot2Wrapper(invSigmaMuDev, featureDev, fDim, cbNum, fStep, resDev, threads2, blocks2, memSize2);
		kernelShareAddCstWrapper(cbNum, fNum, cstDev, resDev, threads3, blocks3);
		err = cudaMemcpy(res + i * cbNum * fStep, resDev, fs * cbNum * sizeof(double), cudaMemcpyDeviceToHost);
	}


	err = cudaFree(resDev);
	err = cudaFree(featureDev);
}

void CUShareCovLh::runWeightedCalcInCuda(double* feature, int fNum, double* res) {
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

	int memSize = fDim * (fDim + 1) / 2 * cStep * sizeof(double) + fStep * fDim * sizeof(double);
	int M = getDeviceProp().sharedMemPerBlock;
	if (memSize > M) {
		printf("process %d codebooks and %d frames in one batch, cost %d bytes of shared memory, more than %d bytes", cStep, fStep, memSize, M);
		exit(-1);
	}

	dim3 threads(fStep, cStep);
	dim3 blocks((cbNum + cStep - 1) / cStep, 1);

	int cStep2 = (M / sizeof(double) - fStep * fDim) / fDim;
	if (cStep2 > 1024 / fStep)
		cStep2 = 1024 / fStep;
	dim3 threads2(fStep, cStep2);
	dim3 blocks2((cbNum + cStep2 - 1) / cStep2, 1);
	int memSize2 = (cStep2 + fStep) * fDim * sizeof(double);

	dim3 threads3(512, 1);
	dim3 blocks3(fStep, 1);

	for (int i = 0; i < fExtendNum / fStep; i++) {

		int fs = fStep;
		if (i == fExtendNum / fStep - 1) {
			fs = fNum - i * fStep;
		}

		err = cudaMemcpy(featureDev, feature + i * fStep * fDim, fs * fDim * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemset(resDev, 0, fStep * cbNum * sizeof(double));
		kernelSharedMvnWrapper(cbNum, fDim, mixNum, invSigmaDev, featureDev, fStep, resDev, threads, blocks, memSize);
		kernelVecDot2Wrapper(invSigmaMuDev, featureDev, fDim, cbNum, fStep, resDev, threads2, blocks2, memSize2);
		kernelShareAddCstWrapper(cbNum, fNum, cstDev, resDev, threads3, blocks3);
		kernelLSEWrapper(cbNum / mixNum, fs, mixNum, alphaDev, resDev, weightedResDev);
		err = cudaMemcpy(res + i * (cbNum / mixNum) * fStep, weightedResDev, fs * (cbNum / mixNum) * sizeof(double), cudaMemcpyDeviceToHost);
	}


	err = cudaFree(resDev);
	err = cudaFree(featureDev);
	err = cudaFree(weightedResDev);
}


void CUShareCovLh::runCalc(double* feature, int fNum, double* res) {
	if (cudaEnabled)
		runCalcInCuda(feature, fNum, res);
	else
		runCalcInC(feature, fNum, res);
}

void CUShareCovLh::runWeightedCalc(double* feature, int fNum, double* res) {
	if (cudaEnabled)
		runWeightedCalcInCuda(feature, fNum, res);
	else
		runWeightedCalcInC(feature, fNum, res);
}