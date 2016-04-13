#include "GMMProbBatchCalc.h"
#include <iostream>
#include "../CommonLib/Math/MathUtility.h"
#include "../CommonLib/commonvars.h"
#include "../CommonLib/CudaCalc/CUGaussLh.h"
#include "../CommonLib/CudaCalc/CUDiagGaussLh.h"
#include "../CommonLib/CudaCalc/CUShareCovLh.h"
#include "../CommonLib/CUMixGaussLh.h"
#define PI 3.14159265358979
#define  DEFAULT_DIM 45


GMMProbBatchCalc::GMMProbBatchCalc(const GMMCodebookSet* cb, bool useCuda, bool useSegmentModel,double coef) {
	codebooks = cb;
	mask = NULL;
	alphaWeightedStateLh = NULL;
	preDurLh = NULL;
	lhBufLen = 0;
	durWeight = 0;
	m_dCoef = coef;
	unmaskedCalc = NULL;
	this->useCuda = useCuda;
	this->useSegmentModel = useSegmentModel;
}

GMMProbBatchCalc::~GMMProbBatchCalc() {

	if (alphaWeightedStateLh != NULL)
		delete [] alphaWeightedStateLh;
	if (preDurLh != NULL)
		delete [] preDurLh;
	if (unmaskedCalc != NULL)
		delete unmaskedCalc;

}

double GMMProbBatchCalc::logOfSumExp(double* componentLh, double* alpha, int n) {
	if (n < 1) {
		std::cout << "sum length should >= 1";
		exit(-1);
	}

	double* temp = new double[n];

	double bestLh = log(alpha[0]) + componentLh[0];
	temp[0] = bestLh;
	int bestIdx = 0;
	for (int i = 1; i < n; i++) {
		double t = log(alpha[i]) + componentLh[i];
		if (t > bestLh) {
			bestLh = t;
			bestIdx = i;
		}
		temp[i] = t;
	}

	for (int i = 0; i < n; i++) {
		temp[i] -= bestLh;
	}

	for (int i = 0; i < n; i++) {
		temp[i] = exp(temp[i]);
	}

	double sum = 0;
	for (int i = 0; i < n; i++) {
		sum += temp[i];
	}

	double res = bestLh + log(sum);
	delete [] temp;

	return res;
}


extern "C" void logSumExpForGMM(int cbNum, int mixNum, int fNum, double* allAlpha, double* seperateLh, double* combinedLh);

void GMMProbBatchCalc::setCodebookSet(const GMMCodebookSet* cb) {
	codebooks = cb;
}

double GMMProbBatchCalc::gausslh(double* x, double* mean, double* invsigma, double cst)
{
	int FDim = codebooks->FDim;
	bool isFullRank = codebooks->isFullRank();
	double res = cst;
	double* t = new double[FDim];
	for (int i = 0; i < FDim; i++)
		t[i] = x[i] - mean[i];

	if (isFullRank) {
		for (int i = 0; i < FDim; i++) {
			double rest = 0;
			for (int j = 0; j < i; j++) 
				rest += invsigma[i * FDim + j] * t[j];
			rest += invsigma[i * FDim + i] * t[i] / 2;
			res += rest * t[i];
		}
	} else {
		double rest = 0;
		for (int i = 0; i < FDim; i++) {
			rest += t[i] * t[i] * invsigma[i];
		}
		res += rest / 2;
	}


	delete [] t;
	return -res;
}

void GMMProbBatchCalc::preparePreDurLh() {
	if (!useSegmentModel) return;

	if (preDurLh != NULL)
		delete [] preDurLh;
	int cbNum = codebooks->CodebookNum;
	preDurLh = new double[cbNum * PRECALCULATED_DURATION_LENGTH];
	for (int i = 0; i < cbNum; i++) {
		if (mask == NULL || mask[i] == true) {
			for (int j = 0; j < PRECALCULATED_DURATION_LENGTH; j++) {
				double durLh = normlh(j + 1, codebooks->DurMean[i], codebooks->DurVar[i], codebooks->LogDurVar[i]) * durWeight;
				preDurLh[i * PRECALCULATED_DURATION_LENGTH + j] = durLh;
			}
		}
	}

}

void GMMProbBatchCalc::setMask(bool* mask) {
	if (this->mask != NULL)
		delete [] this->mask;
	int cbNum = codebooks->CodebookNum;
	this->mask = new bool[cbNum];
	memcpy(this->mask, mask, cbNum * sizeof(bool));
}

void GMMProbBatchCalc::prepare(double* features, int fnum) {
	if (alphaWeightedStateLh != NULL)
		delete [] alphaWeightedStateLh;

	int codebookNum = codebooks->CodebookNum;
	int mixNum = codebooks->MixNum;
	int fDim = codebooks->FDim;
	int cbType = codebooks->cbType;

	lhBufLen = fnum * codebookNum;
	alphaWeightedStateLh = new double[lhBufLen];

	if (mask != NULL) {
		int* maskIdx = new int[codebookNum];

		//count what codebooks will be used in this batch, saving some CPU time
		int maskedCodebookNum = 0;
		for (int i = 0; i < codebookNum; i++) {
			if (mask[i]) {
				maskIdx[maskedCodebookNum] = i;
				maskedCodebookNum++;
			}
		}

		//the const of gaussian distribution, which equlas -ln(Z) 
		//(Z is the normalization constant of a normal distribution)
		//select only codebook allowed by mask

		double* mCst = new double[mixNum * maskedCodebookNum];
		double c0 = fDim / 2.0 * log(2 * PI);
		if (m_dCoef<1e-6 && fDim >= DEFAULT_DIM)
		{
			c0 = DEFAULT_DIM / 2.0 * log(2 * PI);
		}
		for (int i = 0; i < maskedCodebookNum; i++) {
			for (int j = 0; j < mixNum; j++) {
				int p = cbType == GMMCodebookSet::CB_TYPE_FULL_RANK_SHARE ? maskIdx[i] : maskIdx[i] * mixNum + j;

				mCst[i * mixNum + j] = c0 + 0.5 * log(codebooks->DetSigma[p]);
			}
		}

		double* mAlpha = new double[mixNum * maskedCodebookNum];
		for (int i = 0; i < maskedCodebookNum; i++)
			memcpy(mAlpha + i * mixNum, codebooks->Alpha + maskIdx[i] * mixNum, mixNum * sizeof(double));

		int L = mixNum * fDim;
		double* mMu = new double[maskedCodebookNum * L];
		for (int i = 0; i < maskedCodebookNum; i++)
			memcpy(mMu + i * L, codebooks->Mu + maskIdx[i] * L, L * sizeof(double));

		L = codebooks->SigmaL();
		double* mInvSigma = new double[maskedCodebookNum * L];
		for (int i = 0; i < maskedCodebookNum; i++)
			memcpy(mInvSigma + i * L, codebooks->InvSigma + maskIdx[i] * L, L * sizeof(double));

		double* mRes = new double[fnum * maskedCodebookNum];

		if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK) {
			CUGaussLh* cucp = new CUGaussLh(mixNum * maskedCodebookNum, fDim, mInvSigma, mMu, mCst, mAlpha, mixNum, useCuda, m_dCoef);
			cucp->runWeightedCalc(features, fnum, mRes);
			delete cucp;
		} else if (cbType == GMMCodebookSet::CB_TYPE_DIAG) {
			CUDiagGaussLh* cucp = new CUDiagGaussLh(mixNum * maskedCodebookNum, fDim, mInvSigma, mMu, mCst, mAlpha, mixNum, useCuda);
			cucp->runWeightedCalc(features, fnum, mRes);
			delete cucp;
		} else if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK_SHARE) {
			CUShareCovLh* cucp = new CUShareCovLh(mixNum * maskedCodebookNum, fDim, mixNum, mInvSigma, mMu, mCst, mAlpha, useCuda);
			cucp->runWeightedCalc(features, fnum, mRes);
			delete cucp;
		}else if (cbType == GMMCodebookSet::CB_TYPE_FULL_MIX){
			CUMixGaussLh* cucp = new CUMixGaussLh(mixNum * maskedCodebookNum,fDim,mInvSigma,mMu,mAlpha,mixNum,useCuda,codebooks->Beta,codebooks->BetaNum);
			cucp->runWeightedCalc(features, fnum, mRes);
			delete cucp;
		}

		for (int i = 0; i < fnum; i++) {
			for (int j = 0; j < maskedCodebookNum; j++) {
				double t = mRes[i * maskedCodebookNum + j];
				alphaWeightedStateLh[i * codebookNum + maskIdx[j]] = t;
			}
		}
		delete [] mCst;
		delete [] maskIdx;
		delete [] mAlpha;
		delete [] mMu;
		delete [] mInvSigma;
		delete [] mRes;
	}
	else
	{
		if (unmaskedCalc == NULL) {
			if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK) {
				double* Cst = new double[mixNum * codebookNum];
				double c0 = fDim / 2.0 * log(2 * PI);
				if (m_dCoef<=1e-6 && fDim >= DEFAULT_DIM)
				{
					c0 = DEFAULT_DIM / 2.0 * log(2 * PI);
				}
				for (int i = 0; i < codebookNum * mixNum; i++) {
					Cst[i] = c0 + 0.5 * log(codebooks->DetSigma[i]);
				}

				unmaskedCalc = new CUGaussLh(mixNum * codebookNum, fDim, codebooks->InvSigma, codebooks->Mu, Cst, codebooks->Alpha, mixNum, useCuda, m_dCoef);
				delete [] Cst;
			} else if (cbType == GMMCodebookSet::CB_TYPE_DIAG) {
				unmaskedCalc = new CUDiagGaussLh(mixNum * codebookNum, fDim, codebooks->InvSigma, codebooks->Mu, codebooks->Alpha, mixNum, useCuda);
			} else if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK_SHARE) {
				unmaskedCalc = new CUShareCovLh(mixNum * codebookNum, fDim, mixNum, codebooks->InvSigma, codebooks->Mu, codebooks->Alpha, useCuda);
			} else if (cbType == GMMCodebookSet::CB_TYPE_FULL_MIX)
				unmaskedCalc = new CUMixGaussLh(mixNum * codebookNum, fDim, codebooks->InvSigma, codebooks->Mu, codebooks->Alpha, mixNum, useCuda,codebooks->Beta,codebooks->BetaNum);

		}
		if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK)
			((CUGaussLh*)unmaskedCalc)->runWeightedCalc(features, fnum, alphaWeightedStateLh);
		else if (cbType == GMMCodebookSet::CB_TYPE_DIAG)
			((CUDiagGaussLh*)unmaskedCalc)->runWeightedCalc(features, fnum, alphaWeightedStateLh);
		else if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK_SHARE)
			((CUShareCovLh*)unmaskedCalc)->runWeightedCalc(features, fnum, alphaWeightedStateLh);
		else if (cbType == GMMCodebookSet::CB_TYPE_FULL_MIX)
			((CUMixGaussLh*)unmaskedCalc)->runWeightedCalc(features, fnum, alphaWeightedStateLh);
	}

	if (useSegmentModel)
		preparePreDurLh();

}

void GMMProbBatchCalc::setDurWeight(double w) {
	this->durWeight = w;
}

double GMMProbBatchCalc::getDurWeight() {
	return durWeight;
}

void GMMProbBatchCalc::calcSimularity(double *features, int fnum, std::vector<double>& vec)
{
	if(!vec.empty())vec.clear();
	int codebookNum = codebooks->CodebookNum;
	int mixNum = codebooks->MixNum;
	int fDim = codebooks->FDim;
	int cbType = codebooks->cbType;

	auto aWSLh = new double[fnum * codebookNum];
	double* res = new double[codebookNum];
	void* pCUGauss;
	if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK) {
		double* Cst = new double[mixNum * codebookNum];
		double c0 = fDim / 2.0 * log(2 * PI);

		for (int i = 0; i < codebookNum * mixNum; i++) {
			Cst[i] = c0 + 0.5 * log(codebooks->DetSigma[i]);
		}
		pCUGauss = new CUGaussLh(mixNum * codebookNum, fDim, codebooks->InvSigma, codebooks->Mu, Cst, codebooks->Alpha, mixNum, useCuda, m_dCoef);
		delete [] Cst;
	} else if (cbType == GMMCodebookSet::CB_TYPE_DIAG) {
		pCUGauss = new CUDiagGaussLh(mixNum * codebookNum, fDim, codebooks->InvSigma, codebooks->Mu, codebooks->Alpha, mixNum, useCuda);
	} else if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK_SHARE) {
		pCUGauss = new CUShareCovLh(mixNum * codebookNum, fDim, mixNum, codebooks->InvSigma, codebooks->Mu, codebooks->Alpha, useCuda);
	} 


	if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK){
		((CUGaussLh*)pCUGauss)->runWeightedCalc(features, fnum, aWSLh);
		delete (CUGaussLh*)pCUGauss;
	}
	else if (cbType == GMMCodebookSet::CB_TYPE_DIAG){
		((CUDiagGaussLh*)pCUGauss)->runWeightedCalc(features, fnum, aWSLh);
		delete (CUDiagGaussLh*)pCUGauss;
	}
	else if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK_SHARE){
		((CUShareCovLh*)pCUGauss)->runWeightedCalc(features, fnum, aWSLh);
		delete (CUShareCovLh*)pCUGauss;
	}
	for (int i = 0; i != codebookNum; i++)
	{
		res [i] = 0; 
		for (int j = 0; j != fnum; j++)
		{
			res [i]+= aWSLh[i + j * codebookNum];
		}
		res[i] /= fnum;
	}
	
	double gen = MathUtility::logSumExp(res,codebookNum);
	for (int i = 0; i != codebookNum; i++)
	{
		vec.push_back(res[i] - gen);//res[i] = abs(res[i] - gen) > 306 ? 0 : exp(res[i] - gen);
		/*if (res[i]>0.000001)
		{
			vec.push_back(1);
		}
		else{
			vec.push_back(0);
		}*/
	}

	delete []aWSLh;
	delete []res;
}

void GMMProbBatchCalc::CalcAlphaWeightLh(double *features, int fnum,double * res, double* resGen){

	auto aWSLh = res;
	int codebookNum = codebooks->CodebookNum;
	int mixNum = codebooks->MixNum;
	int fDim = codebooks->FDim;
	int cbType = codebooks->cbType;

	void* pCUGauss;
	if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK) {
		double* Cst = new double[mixNum * codebookNum];
		double c0 = fDim / 2.0 * log(2 * PI);

		for (int i = 0; i < codebookNum * mixNum; i++) {
			Cst[i] = c0 + 0.5 * log(codebooks->DetSigma[i]);
		}
		pCUGauss = new CUGaussLh(mixNum * codebookNum, fDim, codebooks->InvSigma, codebooks->Mu, Cst, codebooks->Alpha, mixNum, useCuda, m_dCoef);
		delete [] Cst;
	} else if (cbType == GMMCodebookSet::CB_TYPE_DIAG) {
		pCUGauss = new CUDiagGaussLh(mixNum * codebookNum, fDim, codebooks->InvSigma, codebooks->Mu, codebooks->Alpha, mixNum, useCuda);
	} else if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK_SHARE) {
		pCUGauss = new CUShareCovLh(mixNum * codebookNum, fDim, mixNum, codebooks->InvSigma, codebooks->Mu, codebooks->Alpha, useCuda);
	} 


	if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK){
		((CUGaussLh*)pCUGauss)->runWeightedCalc(features, fnum, aWSLh);
		delete (CUGaussLh*)pCUGauss;
	}
	else if (cbType == GMMCodebookSet::CB_TYPE_DIAG){
		((CUDiagGaussLh*)pCUGauss)->runWeightedCalc(features, fnum, aWSLh);
		delete (CUDiagGaussLh*)pCUGauss;
	}
	else if (cbType == GMMCodebookSet::CB_TYPE_FULL_RANK_SHARE){
		((CUShareCovLh*)pCUGauss)->runWeightedCalc(features, fnum, aWSLh);
		delete (CUShareCovLh*)pCUGauss;
	}

	if (resGen)
	{
		for (int i = 0; i != fnum;i++)
		{
			resGen[i] = MathUtility::logSumExp(aWSLh + i * codebookNum, codebookNum);
		}
	}
}
