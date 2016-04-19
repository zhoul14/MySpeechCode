#include "GMMEstimator.h"
#include "../CommonLib/Math/MathUtility.h"
#include <iostream>
#include "../CommonLib/CudaCalc/CUGaussLh.h"
#include "../CommonLib/CudaCalc/CUDiagGaussLh.h"
#include "../CommonLib/CudaCalc/CUShareCovLh.h"
#include "../CommonLib/CUMixGaussLh.h"
#include <string>
#include <windows.h>
//#include "boost/filesystem.hpp"

GMMEstimator::~GMMEstimator() {
	if (alpha != NULL)
		delete [] alpha;
	if (mu != NULL)
		delete [] mu;
	if (invSigma != NULL)
		delete [] invSigma;
	if (normc != NULL)
		delete [] normc;
	if (data != NULL)
		delete [] data;
	if (precalcLh != NULL)
		delete [] precalcLh;
	if (precalcBetaLh != NULL)
	{
		delete [] precalcBetaLh;
	}
	if (beta != NULL)
	{
		delete [] beta;
	}
}

GMMEstimator::GMMEstimator(int fDim, int mixNum, int cbType, int maxIter, bool useCuda, double coef, int betaNum) 
{
	init(fDim, mixNum, cbType, maxIter, useCuda);
	m_dCoef = coef;
	beta = new double [mixNum * betaNum];

}

GMMEstimator::GMMEstimator(int fDim, int mixNum, int cbType, int maxIter, bool useCuda)
{
	init(fDim, mixNum, cbType, maxIter, useCuda);
	betaNum = 0;
	beta = NULL;
	m_dCoef = 1;
}

void GMMEstimator::init(int fDim, int mixNum, int cbType, int maxIter, bool useCuda)
{
	this->fDim = fDim;
	this->mixNum = mixNum;
	//this->isFullRank = fullRankFlag;
	this->maxIter = maxIter;
	this->cbType = cbType;
	cbId = 0;
	data = NULL;
	logFile = NULL;
	this->useCuda = useCuda;
	//shareCovFlag = SHARE_NONE;
	int L = sigmaL();
	alpha = new double[mixNum];
	mu = new double[fDim * mixNum];
	invSigma = new double[L * mixNum];
	int M = cbType == CB_TYPE_FULL_RANK_SHARE ? 1 : mixNum;
	normc = new double[M];
	precalcLh = NULL;	
	precalcBetaLh = NULL;
}

void GMMEstimator::updateNormC() {

	int L = sigmaL();

	double* m = new double[L];
	for (int i = 0; i < mixNum; i++) {
		memcpy(m, invSigma + i * L, sizeof(double) * L);
		double det = 0;
		if (isFullRank())
			det = MatrixInv(m, fDim);
		else {
			det = MathUtility::prod(m, fDim);
		}

		normc[i] = sqrt(det) / pow(2 * PI, (double)fDim / 2.0);
		if (shareInvSigma())
			break;
	}


	delete [] m;
}

void GMMEstimator::loadParam(double* alpha, double* mu, double* invSigma,double *beta) {

	memcpy(this->alpha, alpha, mixNum * sizeof(double));
	memcpy(this->mu, mu, mixNum * fDim * sizeof(double));
	if (beta)
	{
		memcpy(this->beta,beta,mixNum * 2 *sizeof(double));
		betaNum = 2;
	}
	if (cbType == CB_TYPE_FULL_MIX)
	{
		memcpy(this->invSigma,invSigma,sigmaL() * sizeof(double));
	}
	else if (!shareInvSigma())
		memcpy(this->invSigma, invSigma, mixNum * sigmaL() * sizeof(double));
	else
		memcpy(this->invSigma, invSigma, sigmaL() * sizeof(double));
	//updateNormC();

}

void GMMEstimator::loadData(double* x, int dataNum) {
	this->dataNum = dataNum;
	if (data != NULL)
		delete [] data;
	data = new double[dataNum * fDim];
	memcpy(data, x, dataNum * fDim * sizeof(double));

}

void GMMEstimator::saveParam(double* alpha, double* mu, double* invSigma, double* beta) {
	memcpy(alpha, this->alpha, mixNum * sizeof(double));
	memcpy(mu, this->mu, mixNum * fDim * sizeof(double));
	if (cbType == CB_TYPE_FULL_MIX)
	{
		memcpy(beta, this->beta, betaNum * sizeof(double) * mixNum);
	}
	if (!shareInvSigma())
		memcpy(invSigma, this->invSigma, mixNum * sigmaL() * sizeof(double));
	else
		memcpy(invSigma, this->invSigma, sigmaL() * sizeof(double));
	return;
}

void GMMEstimator::dumpData() {
	//	using namespace boost::filesystem;
	char* dir = "dump";
	// 	if (!exists(dir)) {
	// 		create_directory(dir);
	// 	}
	if (GetFileAttributes(dir) == INVALID_FILE_ATTRIBUTES) {
		CreateDirectory(dir, NULL);
	}
	char filename[20];
	sprintf(filename, "%s/%04d.txt", dir, cbId);
	dumpData(filename);
}

void GMMEstimator::dumpData(char* filename) {
	FILE* fid = fopen(filename, "w");

	fprintf(fid, "FDIM\n%d\n\n", fDim);
	fprintf(fid, "MIXNUM\n%d\n\n", mixNum);

	fprintf(fid, "ALPHA\n");
	for (int i = 0; i < mixNum; i++) {
		fprintf(fid, "%.10f ", alpha[i]);
	}
	fprintf(fid, "\n\n");

	fprintf(fid, "MU\n");
	for (int i = 0; i < fDim * mixNum; i++) {
		fprintf(fid, "%.10f ", mu[i]);
	}
	fprintf(fid, "\n\n");

	fprintf(fid, "INVSIGMA\n");
	for (int i = 0; i < sigmaL() * mixNum; i++) {
		fprintf(fid, "%.10f ", invSigma[i]);
	}
	fprintf(fid, "\n\n");

	fprintf(fid, "DATANUM\n");
	fprintf(fid, "%d\n\n", dataNum);


	fprintf(fid, "DATA\n");
	for (int i = 0; i < dataNum; i++) {
		for (int j = 0; j < fDim; j++) {
			fprintf(fid, "%.10f ", data[i * fDim + j]);
		}
		fprintf(fid, "\n");
	}


	fclose(fid);

}
//改写precalcLh，为下面的更新做预计算
void GMMEstimator::prepareLhForUpdate() {
	if (precalcLh != NULL) {
		delete [] precalcLh;
	}

	precalcLh = new double[dataNum * mixNum];
	memset(precalcLh, 0, dataNum * mixNum * sizeof(double));

	if (cbType == CB_TYPE_FULL_RANK) {
		CUGaussLh* cugl = new CUGaussLh(mixNum, fDim, invSigma, mu, NULL, -1, useCuda, m_dCoef);
		cugl->runCalc(data, dataNum, precalcLh);
		delete cugl;
	} else if (cbType == CB_TYPE_DIAG) {
		CUDiagGaussLh* cugl = new CUDiagGaussLh(mixNum, fDim, invSigma, mu, NULL, -1, useCuda);
		cugl->runCalc(data, dataNum, precalcLh);
		delete cugl;
	} else if (CB_TYPE_FULL_RANK_SHARE) {
		CUShareCovLh* cugl = new CUShareCovLh(mixNum, fDim, mixNum, invSigma, mu, NULL, useCuda);
		cugl->runCalc(data, dataNum, precalcLh);
		delete cugl;
	}

}

bool GMMEstimator::singleGaussianUpdate() {
	if (mixNum != 1) {
		printf("mixNum(%d) != 1, single gaussian update shouldn't be called\n", mixNum);
		exit(-1);
	}
	if (cbType == CB_TYPE_FULL_MIX)
	{
		auto info=updateMix();
		if (info.status == SUCCESS)
		{
			return true;
		}
		return false;
		
	}

	int L = sigmaL();
	double* tempMu = new double[fDim];
	double* tempInvSigma = new double[L];
	double* f = new double[fDim];
	memset(tempMu, 0, fDim * sizeof(double));
	memset(tempInvSigma, 0, L * sizeof(double));

	for (int i = 0; i < dataNum; i++) {
		for (int j = 0; j < fDim; j++) {
			tempMu[j] += data[i * fDim + j];
			if(_isnan(tempMu[j])||!_finite(tempMu[j])){
				dumpData();
				return false;
			}
		}
	}

	for (int i = 0; i < fDim; i++) {
		tempMu[i] /= dataNum;
	}

	for (int i = 0; i < dataNum; i++) {
		for (int j = 0; j < fDim; j++) {
			f[j] = data[i * fDim + j] - tempMu[j];
		}

		if (cbType != CB_TYPE_DIAG) {
			for (int j = 0; j < fDim; j++) {
				for (int k = 0; k <= j; k++) {					
					if ( j < DefaultFdim || k >= DefaultFdim)
					{
						tempInvSigma[j * fDim + k] += f[j] * f[k];
					}
				}
			}
		} else {
			for (int j = 0; j < fDim; j++) {
				tempInvSigma[j] += f[j] * f[j];
			}
		}

	}

	if (cbType != CB_TYPE_DIAG) {
		for (int i = 0; i < fDim; i++) {
			for (int j = 0; j <= i; j++) {
				tempInvSigma[i * fDim + j] /= dataNum;
			}
		}

		for (int i = 0; i < fDim; i++) {
			for (int j = i + 1; j < fDim; j++) {
				tempInvSigma[i * fDim + j] = tempInvSigma[j * fDim + i];
			}
		}
	} else {
		for (int i = 0; i < fDim; i++) {
			tempInvSigma[i] /= dataNum;
		}
	}
	delete [] f;

	if (isFullRank()) {
		if (MathUtility::rcond(tempInvSigma, fDim) < 1e-10) {

			delete [] tempInvSigma;
			delete [] tempMu;
			return false;
		}

		MathUtility::inv(tempInvSigma, fDim);
	} else {
		for (int i = 0; i < fDim; i++) {
			tempInvSigma[i] = 1 / tempInvSigma[i];
			if(tempInvSigma[i]!=tempInvSigma[i]){
				dumpData();
				return false;
			}
		}
	}


	memcpy(mu, tempMu, fDim * sizeof(double));
	memcpy(invSigma, tempInvSigma, L * sizeof(double));
	alpha[0] = 1;

	delete [] tempInvSigma;
	delete [] tempMu;
	return true;
}

std::string GMMEstimator::errorInfo(int e) {
	std::string t;
	if (e == SUCCESS) {
		t = "SUCCESS";
	} else if (e == SAMPLE_NOT_ENOUGH) {
		t = "SAMPLE_NOT_ENOUTH";
	} else if (e == ALPHA_NEAR_ZERO) {
		t = "ALPHA_NEAR_ZERO";
	} else if (e == ILL_CONDITIONED) {
		t = "ILL_CONDITIONED";
	} else {
		t = "UNKNOWN_ERROR";
	}
	return t;
}

void GMMEstimator::setCbId(int id) {
	cbId = id;
}

void GMMEstimator::setOutputFile(FILE* fid) {
	logFile = fid;
}

int GMMEstimator::estimate() {

	if (dataNum < fDim * mixNum / 2) {
		return SAMPLE_NOT_ENOUGH;
	}

	if (mixNum == 1) {
		return singleGaussianUpdate() ? SUCCESS : ILL_CONDITIONED;
	}
	double lastLh = 0;
	for (int i = 0; i < maxIter; i++) {
		UpdateInfo info = updateOnce();

		if (info.status != SUCCESS) {
			dumpData();
			return info.status;
		}

		double lh = info.lhBeforeUpdate;
		printf("iter %d, lh before update = %f\n", i + 1, lh);
		if (logFile)
			fprintf(logFile, "iter %d, lh before update = %f\n", i + 1, lh);
		if (i > 0 && abs((lh - lastLh) / lastLh) < 1e-6) {
			return SUCCESS;
		}
		lastLh = lh;
	}
	return SUCCESS;

}

bool GMMEstimator::isAlphaTooSmall(double* alpha) {
	for (int i = 0; i < mixNum; i++) {
		if (alpha[i] < 1.0 / mixNum / 10)
			return true;
	}
	return false;
}

bool GMMEstimator::isRcondTooSmall(double* mat) {
	double rcond = 0;
	if (cbType != CB_TYPE_DIAG) {
		rcond = MathUtility::rcond(mat, fDim);
		return rcond < 1e-10;
	}
	return false;

}

void GMMEstimator::splitMixtureByAlpha(int idx) {
	const double splitOffset = 0.5;

	double maxAlpha = 0;
	int maxAlphaIdx = -1;
	for (int i = 0; i < mixNum; i++) {
		if (alpha[i] > maxAlpha) {
			maxAlpha = alpha[i];
			maxAlphaIdx = i;
		}
	}
	printf("try reinit mixture %d by spliting mixture %d\n", idx, maxAlphaIdx);
	if (logFile)
		fprintf(logFile, "try reinit mixture %d by spliting mixture %d\n", idx, maxAlphaIdx);

	alpha[maxAlphaIdx] = (alpha[maxAlphaIdx] + alpha[idx]) / 2;
	alpha[idx] = alpha[maxAlphaIdx];

	int L = sigmaL();
	memcpy(mu + fDim * idx, mu + fDim * maxAlphaIdx, fDim * sizeof(double));
	if (cbType != CB_TYPE_FULL_RANK_SHARE)
		memcpy(invSigma + L * idx, invSigma + L * maxAlphaIdx, L * sizeof(double));


	if (cbType != CB_TYPE_DIAG) {
		double eigval = 0;
		double* eigvec = new double[fDim];

		if (cbType == CB_TYPE_FULL_RANK)
			MathUtility::smallest_eig_sym(invSigma + L * maxAlphaIdx, fDim, &eigval, eigvec);
		else
			MathUtility::smallest_eig_sym(invSigma, fDim, &eigval, eigvec);
		//double maxSigma = sqrt(1 / eigval);

		for (int k = 0; k < fDim; k++) {
			mu[fDim * idx + k] += splitOffset * eigvec[k];
		}

		for (int k = 0; k < fDim; k++) {
			mu[fDim * maxAlphaIdx + k] -= splitOffset * eigvec[k];
		}

		delete [] eigvec;
	} else {
		int p = -1;
		double minIS = 0;
		for (int k = 0; k < fDim; k++) {
			if (p < 0 || invSigma[L * maxAlphaIdx + k] < minIS) {
				p = k;
				minIS = invSigma[L * maxAlphaIdx + k];
			}
		}
		mu[fDim * idx + p] += splitOffset;
		mu[fDim * maxAlphaIdx + p] -= splitOffset;
	}
}

UpdateInfo GMMEstimator::updateOnce() {
FUNC_HEAD:
	prepareLhForUpdate();

	UpdateInfo info;
	info.status = SUCCESS;

	//E-step gamma calculate
	double* gamma = new double[dataNum * mixNum];
	double* componentLh = new double[mixNum];
	double maxTlh;
	double lh = 0;
	for (int i = 0; i < dataNum; i++) {
		double z = 0;

		for (int j = 0; j < mixNum; j++) {			
			double tlh = precalcLh[i * mixNum + j];
			double t = exp(tlh) * alpha[j];
			gamma[i * mixNum + j] = t;
			componentLh[j] = alpha[j] > 0 ? tlh + log(alpha[j]) : 0;
			z += t;
		}

		for (int j = 0; j < mixNum; j++) {
			gamma[i * mixNum + j] /= z;
		}
		if(z==0){
			double tempT = precalcLh[i * mixNum + 0];
			for (int j = 0; j< mixNum; j++){
				tempT = tempT > precalcLh[i * mixNum + j] ? tempT : precalcLh[i * mixNum + j];
			}
			for (int j = 0; j < mixNum; j++) {
				double tlh = precalcLh[i * mixNum + j];

				double t = exp(tlh-tempT) * alpha[j];
				gamma[i * mixNum + j] = t;
				componentLh[j] = alpha[j] > 0 ? tlh + log(alpha[j]) : 0;
				z += t;
			}
			for (int j = 0; j < mixNum; j++) {
				gamma[i * mixNum + j] /= z;
			}
		}

		int idx = 0;
		while (alpha[idx] == 0) {idx++;}
		double t = componentLh[idx];
		for (int j = idx + 1; j < mixNum; j++) {
			if (alpha[j] == 0)
				continue;

			double q = componentLh[j];
			if (t > q)
				t = t + log(1 + exp(q - t));
			else
				t = q + log(1 + exp(t - q));
		}
		lh += t;
	}
	info.lhBeforeUpdate = lh;
	delete [] componentLh;


	//记录每个mixture分配到的软样本数
	double* n = new double[mixNum];
	for (int i = 0; i < mixNum; i++) {
		n[i] = 0;
	}
	for (int i = 0; i < mixNum; i++) {
		for (int j = 0; j < dataNum; j++) {
			n[i] += gamma[j * mixNum + i];
		}
	}

	//M-step, 更新alpha
	for (int i = 0; i < mixNum; i++) {
		alpha[i] = n[i] / dataNum;
	}
	if (logFile) {
		fprintf(logFile, "alpha: [");
		for (int i = 0; i < mixNum; i++) {
			fprintf(logFile, "%f, ", alpha[i]);
		}
		fprintf(logFile, "]\n");
	}


	bool reinit = false;
	for (int i = 0; i < mixNum; i++) {
		if (n[i] < 10) {
			reinit = true;
			splitMixtureByAlpha(i);
			if (logFile) {
				fprintf(logFile, "after split alpha: [");
				for (int i = 0; i < mixNum; i++) {
					fprintf(logFile, "%f, ", alpha[i]);
				}
				fprintf(logFile, "]\n");
			}

		}
	}

	if (reinit) {
		goto FUNC_HEAD;
	}


	//M-step, 更新mu
	for (int i = 0; i < mixNum; i++) {
		double* tempMu = new double[fDim];
		memset(tempMu, 0, fDim * sizeof(double));
		for (int j = 0; j < dataNum; j++) {
			for (int k = 0; k < fDim; k++) {
				tempMu[k] += gamma[j * mixNum + i] * data[j * fDim + k];
			}
		}

		for (int j = 0; j < fDim; j++) {
			tempMu[j] /= n[i];
		}

		memcpy(mu + i * fDim, tempMu, fDim * sizeof(double));
		delete [] tempMu;
	}


	//M-step, 更新invSigma
	int L = sigmaL();
	double* sharedInvSigma = NULL;
	if (shareInvSigma()) {
		sharedInvSigma = new double[fDim * fDim];
		memset(sharedInvSigma, 0, fDim * fDim * sizeof(double));
	}
	//to be continued

	for (int i = 0; i < mixNum; i++) {
		double* tempInvSigma = new double[L];
		memset(tempInvSigma, 0, L * sizeof(double));

		for (int j = 0; j < dataNum; j++) {
			double* f = new double[fDim];
			for (int k = 0; k < fDim; k++) {
				f[k] = mu[i * fDim + k] - data[j * fDim + k];
			}

			double g = gamma[j * mixNum + i];
			if (isFullRank()) {
				for (int k = 0; k < fDim; k++) {
					for (int l = 0; l <= k; l++) {
						if ( k >= DefaultFdim && l < DefaultFdim)
						{
							l = DefaultFdim - 1;
							continue;
						}
						tempInvSigma[k * fDim + l] += f[k] * f[l] * g;
					}
				}
			} else {
				for (int k = 0; k < fDim; k++) {
					tempInvSigma[k] += f[k] * f[k] * g;
				}
			}

			delete [] f;
		}

		if (isFullRank()) {
			for (int j = 0; j < fDim; j++) {
				for (int k = j + 1; k < fDim; k++) {
					tempInvSigma[j * fDim + k] = tempInvSigma[k * fDim + j];
				}
			}

			for (int j = 0; j < fDim; j++) {
				for (int k = 0; k < fDim; k++) {
					tempInvSigma[j * fDim + k] /= n[i];
				}
			}
		} else {
			for (int j = 0; j < fDim; j++) {
				tempInvSigma[j] /= n[i];
			}
		}

		if (!shareInvSigma() && isRcondTooSmall(tempInvSigma)) {
			delete [] tempInvSigma;
			delete [] n;
			delete [] gamma;
			info.status = ILL_CONDITIONED;
			info.mixIdx = i;
			return info;
		}


		if (shareInvSigma())	{//share covariance matrix
			for (int j = 0; j < fDim * fDim; j++)
				sharedInvSigma[j] += tempInvSigma[j] * alpha[i];
		} else { //no sharing
			if (isFullRank())
				MatrixInv(tempInvSigma, fDim);	//det = |Sigma|
			else {
				for (int j = 0; j < fDim; j++)
					tempInvSigma[j] = 1 / tempInvSigma[j];
			}
		}

		if (!shareInvSigma()) {
			for (int j = 0; j < L; j++) {
				if (tempInvSigma[j] != tempInvSigma[j]) {
					printf("error! nan in EM update\n");
					exit(-1);
				}
			}
			memcpy(invSigma + i * L, tempInvSigma, L * sizeof(double));
		}
		delete [] tempInvSigma;
	}
	delete [] gamma;
	delete [] n;

	if (shareInvSigma()) {	//if share

		if (isRcondTooSmall(sharedInvSigma)) {
			delete [] sharedInvSigma;
			info.status = ILL_CONDITIONED;
			info.mixIdx = 0;
			return info;
		}

		memcpy(invSigma, sharedInvSigma, fDim * fDim * sizeof(double));
		MatrixInv(invSigma, fDim);

	}

	//更新归一化常数
	updateNormC();

	if (shareInvSigma()) {
		delete [] sharedInvSigma;
	}

	return info;

}


UpdateInfo GMMEstimator::updateMix(){
FUNC_HEAD:
	prepareLhForUpdateMIx();

	UpdateInfo info;
	info.status = SUCCESS;

	//E-step
	double* gamma = new double[dataNum * betaNum];
	double* componentLh = new double[betaNum];
	double maxTlh;
	double lh = 0;
	for (int i = 0; i < dataNum; i++) {
		double z = 0;

		for (int j = 0; j < betaNum; j++) {			
			double tlh = precalcBetaLh[i * betaNum + j];
			double t = exp(tlh) * beta[j];
			gamma[i * betaNum + j] = t;
			componentLh[j] = beta[j] > 0 ? tlh + log(beta[j]) : 0;
			z += t;
		}

		for (int j = 0; j < betaNum; j++) {
			gamma[i * betaNum + j] /= z;
		}
		int idx = 0;
		while (beta[idx] == 0) {idx++;}
		double t = componentLh[idx];
		for (int j = idx + 1; j < betaNum; j++) {
			if (beta[j] == 0)
				continue;

			double q = componentLh[j];
			if (t > q)
				t = t + log(1 + exp(q - t));
			else
				t = q + log(1 + exp(t - q));
		}
		lh += t;
	}
	info.lhBeforeUpdate = lh;
	delete [] componentLh;


	//记录每个mixture分配到的软样本数
	double* n = new double[betaNum];
	for (int i = 0; i < betaNum; i++) {
		n[i] = 0;
	}
	for (int i = 0; i < betaNum; i++) {
		for (int j = 0; j < dataNum; j++) {
			n[i] += gamma[j * betaNum + i];
		}
	}

	//M-step, 更新alpha
	for (int i = 0; i < betaNum; i++) {
		beta[i] = n[i] / dataNum;
	}
	if (logFile) {
		fprintf(logFile, "beta: [");
		for (int i = 0; i < betaNum; i++) {
			fprintf(logFile, "%f, ", beta[i]);
		}
		fprintf(logFile, "]\n");
	}


	//M-step, 更新mu
	double* tempMu = new double[fDim];

	memset(tempMu, 0, fDim * sizeof(double));

	for (int i = 0; i < betaNum; i++) {

		if (i == 0){
			for (int j = 0; j < dataNum; j++) { 

				for (int k = 0; k < FEATURE_DIM; k++) {
					tempMu[k] += gamma[j * betaNum + i] * data[j * fDim + k];
				}
			}
			for (int j = 0; j < FEATURE_DIM; j++) {
				tempMu[j] /= n[i];
			}
		}
		else{
			for (int j = 0; j < dataNum; j++) { 

				for (int k = FEATURE_DIM; k < fDim; k++) {
					tempMu[k] += gamma[j * betaNum + i] * data[j * fDim + k];
				}
			}
			for (int j = FEATURE_DIM; j < fDim; j++) {
				tempMu[j] /= n[i];
			}
		}
	}

	memcpy(mu , tempMu, fDim * sizeof(double));

	delete [] tempMu;


	//M-step, 更新invSigma
	int L = FEATURE_DIM * FEATURE_DIM + (fDim - FEATURE_DIM) * (fDim - FEATURE_DIM);
	//to be continued


	double* tempInvSigma = new double[L];
	memset(tempInvSigma, 0, L * sizeof(double));
	double* tempInvSigma2 = tempInvSigma + FEATURE_DIM * FEATURE_DIM; 

	for (int i = 0; i < betaNum; i++) {


		if (i ==0)
		{
			for (int j = 0; j < dataNum; j++) {
				double* f = new double[FEATURE_DIM];
				for (int k = 0; k < FEATURE_DIM; k++) {
					f[k] = mu[k] - data[j * fDim + k];
				}

				double g = gamma[j * betaNum + i];

				for (int k = 0; k < FEATURE_DIM; k++) {
					for (int l = 0; l <= k; l++) {
						tempInvSigma[k * FEATURE_DIM + l] += f[k] * f[l] * g;
					}
				}
				delete [] f;
			}
			for (int j = 0; j < FEATURE_DIM; j++) {
				for (int k = j + 1; k < FEATURE_DIM; k++) {
					tempInvSigma[j * FEATURE_DIM + k] = tempInvSigma[k * FEATURE_DIM + j];
				}
			}
			for (int j = 0; j < FEATURE_DIM; j++) {
				for (int k = 0; k < FEATURE_DIM; k++) {
					tempInvSigma[j * FEATURE_DIM + k] /= n[i];
				}
			}
			MatrixInv(tempInvSigma, FEATURE_DIM);

			if (!shareInvSigma() && MathUtility::rcond(tempInvSigma,FEATURE_DIM) < 1e-10) {
				delete [] tempInvSigma;
				delete [] n;
				delete [] gamma;
				info.status = ILL_CONDITIONED;
				info.mixIdx = i;
				return info;
			}
		}
		else{
			int apFDim = fDim - FEATURE_DIM;

			for (int j = 0; j < dataNum; j++) {
				double* f = new double[apFDim];
				for (int k = 0; k < apFDim; k++) {
					f[k] = mu[k + FEATURE_DIM] - data[j * fDim + FEATURE_DIM + k];
				}

				double g = gamma[j * betaNum + i];

				for (int k = 0; k < apFDim; k++) {
					for (int l = 0; l <= k; l++) {
						tempInvSigma2[k * apFDim + l] += f[k] * f[l] * g;
					}
				}
				delete [] f;
			}
			for (int j = 0; j < apFDim; j++) {
				for (int k = j + 1; k < apFDim; k++) {
					tempInvSigma2[j * apFDim + k] = tempInvSigma2[k * apFDim + j];
				}
			}
			for (int j = 0; j < apFDim; j++) {
				for (int k = 0; k < apFDim; k++) {
					tempInvSigma2[j * apFDim + k] /= n[i];
				}
			}
			MatrixInv(tempInvSigma2, apFDim);
			if (!shareInvSigma() && MathUtility::rcond(tempInvSigma2,apFDim) < 1e-10) {
				delete [] tempInvSigma;
				delete [] n;
				delete [] gamma;
				info.status = ILL_CONDITIONED;
				info.mixIdx = i;
				return info;
			}
		}		
	}

	if (!shareInvSigma()) {
		for (int j = 0; j < L; j++) {
			if (tempInvSigma[j] != tempInvSigma[j]) {
				printf("error! nan in EM update\n");
				exit(-1);
			}
		}
		memcpy(invSigma, tempInvSigma, L * sizeof(double));
	}
	delete [] tempInvSigma;

	delete [] gamma;
	delete [] n;
	
	return info;
}

void GMMEstimator::prepareLhForUpdateMIx(){

	precalcBetaLh = new double[dataNum * betaNum];
	memset(precalcBetaLh, 0, dataNum * betaNum * sizeof(double));

	if (cbType == CB_TYPE_FULL_MIX) {
		CUMixGaussLh* cugl = new CUMixGaussLh(mixNum, fDim, invSigma, mu, NULL, -1, useCuda, beta,2);
		cugl->runCalcInCBeta(data, dataNum, precalcBetaLh);
		delete cugl;
	} 
}
