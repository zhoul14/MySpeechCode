/*#include <cuda_runtime.h>*/
#include <cuda.h>
#include <stdio.h>

__global__ void kernel_vecDotProduct(double* invSigmaMuDev, double* muDev, int fDim, int cbNum, double* resDev) {
	int cbIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (cbIdx < cbNum) {
		double t = 0;
		double* v1 = invSigmaMuDev + cbIdx * fDim;
		double* v2 = muDev + cbIdx * fDim;
		for (int i = 0; i < fDim; i++) {
			t += v1[i] * v2[i];
		}
		resDev[cbIdx] += t / 2;
	}
}

__global__ void kernel_batchedDgemv(double* invSigmaDev, double* muDev, int fDim, int cbNum, int shareNum, double* resDev) {
	extern __shared__ double sharedMu[];

	int N = blockDim.x;
	int vecLen = N * fDim;
	int maxIdx = cbNum * fDim;
	for (int i = 0; i < (vecLen + blockDim.x - 1) / blockDim.x; i++) {
		int sharedIdx = i * blockDim.x + threadIdx.x;
		int devIdx = blockIdx.x * vecLen + sharedIdx;
		
		if (devIdx < maxIdx)
			sharedMu[sharedIdx] = muDev[devIdx];
	}
	__syncthreads();

	int invSigmaLen = (fDim + 1) * fDim / 2;
	int cbIdx = N * blockIdx.x + threadIdx.x;
	if (cbIdx < cbNum) {
		double* r = resDev + cbIdx * fDim;
		double* v = sharedMu + threadIdx.x * fDim;
		double* A = invSigmaDev + (cbIdx / shareNum) * invSigmaLen;
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
}


__global__ void kernel_lse(int mixNum, double* allAlpha, double* seperateLh, double* combinedLh) {
	extern __shared__ double alpha[];

	for (int i = 0; i < (mixNum + blockDim.x - 1) / blockDim.x; i++) {
		if (i * blockDim.x + threadIdx.x < mixNum) {
			alpha[i * blockDim.x + threadIdx.x] = allAlpha[blockIdx.x * mixNum + blockDim.x * i + threadIdx.x];
		}
	}

	__syncthreads();


	double maxLh = 0;
 	int maxLhIdx = -1;
	for (int i = 0; i < mixNum; i++) {
		if (alpha[i] > 0) {
			int p = gridDim.x * mixNum * threadIdx.x + blockIdx.x * mixNum + i;
			double t = log(alpha[i]) + seperateLh[p];
			//printf("seperateLh[%d] = %f\n", p, seperateLh[p]);
			if (maxLhIdx == -1 || t > maxLh) {
				maxLh = t;
				maxLhIdx = i;
			}
			seperateLh[p] = t;
		}
	}



	for (int i = 0; i < mixNum; i++) {
		if (alpha[i] > 0) {
			int p = gridDim.x * mixNum * threadIdx.x + blockIdx.x * mixNum + i;
			seperateLh[p] -= maxLh;
		}
	}


	for (int i = 0; i < mixNum; i++) {
		if (alpha[i] > 0) {
			int p = gridDim.x * mixNum * threadIdx.x + blockIdx.x * mixNum + i;
			seperateLh[p] = exp(seperateLh[p]);	//此时seperateLh[p]一定小于1
		}
	}
	
	double sumExp = 0;
	for (int i = 0; i < mixNum; i++) {
		if (alpha[i] > 0) {
			int p = gridDim.x * mixNum * threadIdx.x + blockIdx.x * mixNum + i;
			sumExp += seperateLh[p];
		}
	}

	double logSumExp = maxLh + log(sumExp);
	combinedLh[gridDim.x * threadIdx.x + blockIdx.x] = logSumExp;


}

extern "C" void logSumExpForGMM(int cbNum, int mixNum, int fNum, double* allAlpha, double* seperateLh, double* combinedLh) {
	int fStep = 128;
	int fExtendNum = ((fNum + fStep - 1) / fStep) * fStep;

	cudaError_t err;

	double* allAlphaDev;
	err = cudaMalloc((void**)&allAlphaDev, cbNum * mixNum * sizeof(double));
	err = cudaMemcpy(allAlphaDev, allAlpha, cbNum * mixNum * sizeof(double), cudaMemcpyHostToDevice);

	double* seperateLhDev;
	err = cudaMalloc((void**)&seperateLhDev, fExtendNum * mixNum * cbNum * sizeof(double));
	err = cudaMemset(seperateLhDev, 0, fExtendNum * mixNum * cbNum * sizeof(double));
	cudaMemcpy(seperateLhDev, seperateLh, cbNum * mixNum * fNum * sizeof(double), cudaMemcpyHostToDevice);

	double* combinedLhDev;
	err = cudaMalloc((void**)&combinedLhDev, fExtendNum * cbNum * sizeof(double));

	//int lastCpyFrameNum = fNum - (fExtendNum - fStep);
	for (int i = 0; i < fExtendNum / fStep; i++) {
		double* lhInDev = seperateLhDev + fStep * mixNum * cbNum * i;
		double* lhOutDev = combinedLhDev + fStep * cbNum * i;
		kernel_lse<<<cbNum, fStep, mixNum * sizeof(double)>>>(mixNum, allAlphaDev, lhInDev, lhOutDev);
		err = cudaDeviceSynchronize();
		if (err != CUDA_SUCCESS) {
			printf("kernel launch failed in LSE with error \"%s\".\n",
			cudaGetErrorString(err));
			exit(-1);
		}
	}

	err = cudaMemcpy(combinedLh, combinedLhDev, fNum * cbNum * sizeof(double), cudaMemcpyDeviceToHost);

//	free(allLogAlpha);
	cudaFree(combinedLhDev);
	cudaFree(seperateLhDev);
	cudaFree(allAlphaDev);
}


__global__ void kernel_mvn_share(int sigmaNum, int fDim, int shareNum, double* invSigma, double* feature, int fNum, double* resBuf)
{	

	extern __shared__ double memory[];

	double* sharedInvSigma = memory;
	int invSigmaLen = fDim * (fDim + 1) / 2;
	int fLen = fNum * fDim;

	double* sharedFt = sharedInvSigma + blockDim.y * invSigmaLen;
	//memset(sharedFt, 0, fDim * fNum * sizeof(double));

	double res = 0;
	int sigmaIdx = blockDim.y * blockIdx.x + threadIdx.y;
	int thIdx = threadIdx.x + threadIdx.y * blockDim.x;
	int thNum = blockDim.x * blockDim.y;

	if (sigmaIdx < sigmaNum)
	{
		for (int i = 0; i < (invSigmaLen + blockDim.x - 1) / blockDim.x; i++)
			if (i * blockDim.x + threadIdx.x < invSigmaLen)
				sharedInvSigma[threadIdx.y * invSigmaLen + i * blockDim.x + threadIdx.x] = invSigma[sigmaIdx * invSigmaLen + i * blockDim.x + threadIdx.x];
	}

	for (int i = 0; i < (fLen + thNum - 1) / thNum; i++)
		if (i * thNum + thIdx < fLen)
			sharedFt[i * thNum + thIdx] = feature[i * thNum + thIdx];

	__syncthreads();

	if (sigmaIdx < sigmaNum) {
		res = 0;
		int idx = 0;
		for (int j = 0; j < fDim; j++) {
			double t1 = sharedFt[threadIdx.x * fDim + j];
			double rest = 0;
			for (int k = 0; k < j; k++) {
				double t2 = sharedFt[threadIdx.x * fDim + k];
				rest += sharedInvSigma[threadIdx.y * invSigmaLen + idx++] * t2;
			}
			rest += sharedInvSigma[threadIdx.y * invSigmaLen + idx++] * t1;
			res += rest * t1;
		}

		int resIdxStart = (sigmaIdx + threadIdx.x * sigmaNum) * shareNum;
		for (int i = 0; i < shareNum; i++) {
			resBuf[resIdxStart + i] -= res;
		}
		//resBuf[resIdx] = -res;
	}
	__syncthreads();
}

__global__ void kernel_share_add_cst(int cbNum, int fNum, double* cst, double* res) {
	double* fVec = res + blockIdx.x * cbNum;
	for (int i = 0; i < (cbNum + blockDim.x - 1) / blockDim.x; i++)
		if (i * blockDim.x + threadIdx.x < cbNum)
			fVec[i * blockDim.x + threadIdx.x] -= cst[i * blockDim.x + threadIdx.x];
}

__global__ void kernel_vecdot2(int cbNum, int fNum, int fDim, double* invSigmaMu, double* fVec, double* resBuf) {
	extern __shared__ double memory[];
	double* sharedFeature = memory;
	double* sharedInvSigmaMu = sharedFeature + fNum * fDim;

	int cbIdx = blockDim.y * blockIdx.x + threadIdx.y;
	int thNum = blockDim.x * blockDim.y;
	int thIdx = threadIdx.y * blockDim.x + threadIdx.x;

	for (int i = 0; i < (fDim * fNum + thNum - 1) / thNum; i++)
		if (i * thNum + thIdx < fDim * fNum)
			sharedFeature[i * thNum + thIdx] = fVec[i * thNum + thIdx];

	if (cbIdx < cbNum) {
		for (int i = 0; i < (fDim + blockDim.x - 1) / blockDim.x; i++)
			if (i * blockDim.x + threadIdx.x < fDim)
				sharedInvSigmaMu[threadIdx.y * fDim + i * blockDim.x + threadIdx.x] = invSigmaMu[cbIdx * fDim + i * blockDim.x + threadIdx.x];
	}


	__syncthreads();

	if (cbIdx < cbNum) {
		int resIdx = threadIdx.x * cbNum + cbIdx;
		double* v1 = sharedFeature + threadIdx.x * fDim;
		double* v2 = sharedInvSigmaMu + threadIdx.y * fDim;
		double res = 0;
		for (int i = 0; i < fDim; i++) {
			res += v1[i] * v2[i];
		}
		resBuf[resIdx] += res;
	}
}

__global__ void kernel_mvn(int cbNum, int fDim, double* invSigma, double* invSigmaMu, double* cst, double* feature, int fNum, double* resBuf)
{	

	extern __shared__ double memory[];

	double* sharedInvSigma = memory;
	int invSigmaLen = fDim * (fDim + 1) / 2;
	int fLen = fNum * fDim;

	double* sharedInvSigmaMu = sharedInvSigma + blockDim.y * invSigmaLen;
	double* sharedCst = sharedInvSigmaMu + blockDim.y * fDim;
	double* sharedFt = sharedCst + blockDim.y;
	//memset(sharedFt, 0, fDim * fNum * sizeof(double));

	double res = 0;
	int cbIdx = blockDim.y * blockIdx.x + threadIdx.y;
	int thIdx = threadIdx.x + threadIdx.y * blockDim.x;
	int thNum = blockDim.x * blockDim.y;

	if (cbIdx < cbNum)
	{
		for (int i = 0; i < (invSigmaLen + blockDim.x - 1) / blockDim.x; i++)
			if (i * blockDim.x + threadIdx.x < invSigmaLen)
				sharedInvSigma[threadIdx.y * invSigmaLen + i * blockDim.x + threadIdx.x] = invSigma[cbIdx * invSigmaLen + i * blockDim.x + threadIdx.x];

		for (int i = 0; i < (fDim + blockDim.x - 1) / blockDim.x; i++)
			if (i * blockDim.x + threadIdx.x < fDim)
				sharedInvSigmaMu[threadIdx.y * fDim + i * blockDim.x + threadIdx.x] = invSigmaMu[cbIdx * fDim + i * blockDim.x + threadIdx.x];
		
		if (threadIdx.x == 0)
			sharedCst[threadIdx.y] = cst[cbIdx];
	}

	for (int i = 0; i < (fLen + thNum - 1) / thNum; i++)
		if (i * thNum + thIdx < fLen)
			sharedFt[i * thNum + thIdx] = feature[i * thNum + thIdx];

	__syncthreads();

	if (cbIdx < cbNum) {
		res = sharedCst[threadIdx.y];
		int idx = 0;
		for (int j = 0; j < fDim; j++) {
			double t1 = sharedFt[threadIdx.x * fDim + j];
			double rest = 0;
			for (int k = 0; k <= j; k++) {
				double t2 = sharedFt[threadIdx.x * fDim + k];
				rest += sharedInvSigma[threadIdx.y * invSigmaLen + idx++] * t2;
			}
			//rest += sharedInvSigma[threadIdx.y * invSigmaLen + idx++] * t1;
			rest -= sharedInvSigmaMu[threadIdx.y * fDim + j];
			res += rest * t1;
		}

		int resIdx = cbIdx + threadIdx.x * cbNum;
		resBuf[resIdx] = -res;
	}
	__syncthreads();
}
//kernel_share_add_cst(int cbNum, int fNum, double* cst, double* res)
extern "C" void kernelShareAddCstWrapper(int cbNum, int fNum, double* cstDev, double* resDev ,dim3 threads, dim3 blocks) {
	kernel_share_add_cst<<< blocks, threads>>>(cbNum, fNum, cstDev, resDev);
	cudaError_t err;
	err = cudaPeekAtLastError();

	if (err != CUDA_SUCCESS) {
		printf("kernel share_add_cst launch failed with error \"%s\".\n",
			cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

extern "C" void kernelVecDotWrapper(double* invSigmaMuDev, double* muDev, int fDim, int cbNum, double* resDev , dim3 threads, dim3 blocks) {
	kernel_vecDotProduct<<< blocks, threads>>>(invSigmaMuDev, muDev, fDim, cbNum, resDev);
	cudaError_t err;
	err = cudaPeekAtLastError();

	if (err != CUDA_SUCCESS) {
		printf("kernel vecdot launch failed with error \"%s\".\n",
			cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

extern "C" void kernelDgemvWrapper(double* invSigmaDev, double* muDev, int fDim, int cbNum, int shareNum, double* resDev , dim3 threads, dim3 blocks, int memSize) {

	kernel_batchedDgemv<<< blocks, threads, memSize>>>(invSigmaDev, muDev, fDim, cbNum, shareNum, resDev);
	
	cudaError_t err = cudaPeekAtLastError();

	if (err != CUDA_SUCCESS) {
		printf("kernel dgemv launch failed with error \"%s\".\n",
			cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

extern "C" void kernelVecDot2Wrapper(double* invSigmaMuDev, double* featureDev, int fDim, int cbNum, int fNum, double* resDev , dim3 threads, dim3 blocks, int memSize) {
	//kernel_vecdot2(int cbNum, int fNum, int fDim, double* invSigmaMu, double* fVec, double* resBuf)
	kernel_vecdot2<<< blocks, threads, memSize>>>(cbNum, fNum, fDim, invSigmaMuDev, featureDev, resDev);
	cudaError_t err;
	err = cudaPeekAtLastError();

	if (err != CUDA_SUCCESS) {
		printf("kernel vecdot2 launch failed with error \"%s\".\n",
			cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

extern "C" void kernelSharedMvnWrapper(int cbNum, int fDim, int mixNum, double* invSigmaDev, double* featureDev, int fNum, double* resBufDev, dim3 threads, dim3 blocks, int memSize) {
	
	kernel_mvn_share<<< blocks, threads, memSize >>>(cbNum / mixNum, fDim, mixNum, invSigmaDev, featureDev, fNum, resBufDev);
	cudaError_t err;
	err = cudaPeekAtLastError();
 	
	if (err != CUDA_SUCCESS) {
		printf("kernel launch failed with error \"%s\".\n",
		cudaGetErrorString(err));
		exit(-1);
	}
	cudaDeviceSynchronize();
	return;
}

extern "C" void kernelMvnWrapper(int cbNum, int fDim, double* invSigmaDev, double* invSigmaMuDev, double* cstDev, double* featureDev, int fNum, double* resBufDev, dim3 threads, dim3 blocks, int memSize) {
	//kernel_mvn(int cbNum, int fDim, double* invSigma, double* invSigmaMu, double* cst, double* feature, int fNum, double* resBuf)
	kernel_mvn<<< blocks, threads, memSize >>>(cbNum, fDim, invSigmaDev, invSigmaMuDev, cstDev, featureDev, fNum, resBufDev);
	cudaError_t err;
	err = cudaPeekAtLastError();

	if (err != CUDA_SUCCESS) {
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(err));
		exit(-1);
	}
	cudaDeviceSynchronize();
	return;
}


__global__ void kernel_diag_mvn(int cbNum, int fDim, double* invSigma, double* mu, double* cst, double* feature, int fNum, double* resBuf)
{	

	extern __shared__ double memory[];

	double* sharedInvSigma = memory;
	int fLen = fNum * fDim;

	double* sharedMu = sharedInvSigma + blockDim.y * fDim;
	double* sharedCst = sharedMu + blockDim.y * fDim;
	double* sharedFt = sharedCst + blockDim.y;

	int cbIdx = blockDim.y * blockIdx.x + threadIdx.y;
	int thIdx = threadIdx.x + threadIdx.y * blockDim.x;
	int thNum = blockDim.x * blockDim.y;

	if (cbIdx < cbNum)
	{
		for (int i = 0; i < (fDim + blockDim.x - 1) / blockDim.x; i++)
			if (i * blockDim.x + threadIdx.x < fDim)
				sharedInvSigma[threadIdx.y * fDim + i * blockDim.x + threadIdx.x] = invSigma[cbIdx * fDim + i * blockDim.x + threadIdx.x];

		for (int i = 0; i < (fDim + blockDim.x - 1) / blockDim.x; i++)
			if (i * blockDim.x + threadIdx.x < fDim)
				sharedMu[threadIdx.y * fDim + i * blockDim.x + threadIdx.x] = mu[cbIdx * fDim + i * blockDim.x + threadIdx.x];

		if (threadIdx.x == 0)
			sharedCst[threadIdx.y] = cst[cbIdx];
	}

	for (int i = 0; i < (fLen + thNum - 1) / thNum; i++)
		if (i * thNum + thIdx < fLen)
			sharedFt[i * thNum + thIdx] = feature[i * thNum + thIdx];

	__syncthreads();

	if (cbIdx < cbNum) {
		double res = 0;
		for (int j = 0; j < fDim; j++) {
			double xj = sharedFt[threadIdx.x * fDim + j] - sharedMu[threadIdx.y * fDim + j];
			res += xj * xj * sharedInvSigma[threadIdx.y * fDim + j];
		}
		res = -(sharedCst[threadIdx.y] + res / 2);

		int resIdx = cbIdx + threadIdx.x * cbNum;
		resBuf[resIdx] = res;
	}
	__syncthreads();
}

extern "C" void kernelDiagMvnWrapper(int cbNum, int fDim, double* invSigmaDev, double* muDev, double* cstDev, double* featureDev, int fNum, double* resBufDev, dim3 threads, dim3 blocks, int memSize) {

	
	kernel_diag_mvn<<< blocks, threads, memSize >>>(cbNum, fDim, invSigmaDev, muDev, cstDev, featureDev, fNum, resBufDev);
	cudaError_t err;
	err = cudaDeviceSynchronize();
	
	if (err != CUDA_SUCCESS) {
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(err));
		exit(-1);
	}
	
	return;
}
//__global__ void kernel_lse(int mixNum, double* allAlpha, double* seperateLh, double* combinedLh)
extern "C" void kernelLSEWrapper(int cbNum, int fStep, int mixNum, double* alphaDev, double* seperateLhDev, double* combineLhDev) {
	kernel_lse<<<cbNum, fStep, mixNum * sizeof(double)>>>(mixNum, alphaDev, seperateLhDev, combineLhDev);
	cudaError_t err;
	err = cudaDeviceSynchronize();

	if (err != CUDA_SUCCESS) {
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

