#include "WordLevelMMIEstimator.h"
#include "../CommonLib/Math//MathUtility.h"
#include <iostream>
#include "windows.h"
#include <stdlib.h> 
#include "math.h"
void CWordLevelMMIEstimator::loadWordGamma(std::vector<double>& WordGamma){
	mWordGamma = WordGamma;
}

int CWordLevelMMIEstimator::estimate(){
	printf("WordLevel Estimator!\n");
	if (dataNum < fDim * mixNum / 2) {
		return SAMPLE_NOT_ENOUGH;
	}
	int res = 0;
	for (int i = 0; i < 1; i++)
	{
		if (mixNum == 1) 
		{
			m_dDsm = -2;
			res = singleGaussianMMIEupdate() ? SUCCESS : ILL_CONDITIONED;
		}
		else
		{
			printf("Fata:Multi Mixture Api have not coding!\n");
			res = ILL_CONDITIONED;
		}
	}
	return res;
}


CWordLevelMMIEstimator::CWordLevelMMIEstimator(int fDim, int mixNum, int cbType, int maxIter, bool useCuda, std::vector<double>& WordGamma): GMMEstimator(fDim, mixNum, cbType, maxIter, useCuda),
mWordGamma( WordGamma),
m_dDsm(-2)
{
}

bool CWordLevelMMIEstimator::singleGaussianMMIEupdate()
{
	if (mixNum != 1) {
		printf("mixNum(%d) != 1, single gaussian update shouldn't be called\n", mixNum);
		exit(-1);
	}

	//precalcMMIEgamma();

	// Mu update
	double* GammaX = new double[fDim];
	double* tempMu = new double[fDim];

	double GammaOne(0.0);
	memset(GammaX, 0, fDim * sizeof(double));
	int MMIElen = mWordGamma.size();
	if (MMIElen != dataNum)
	{
		printf("FATA:WordGamma size:[%d] != dataNum:[%d] .",MMIElen,dataNum);
	}
	double dDsm = 0;
	for (int i = 0 ; i != MMIElen; i++)
	{
		for (int k = 0; k != fDim; k++)
		{
			GammaX[k] += data[i * fDim + k] * mWordGamma[i];
		}
		dDsm += abs(mWordGamma[i]);
		GammaOne += mWordGamma[i];
	}
	dDsm *= 3;
	dDsm = (dDsm > m_dDsm)? dDsm : m_dDsm;
	double Den = GammaOne + dDsm;
	printf("GammaOne:[%lf],dDsm:[%lf],Den:[%lf]",GammaOne,Den);
	//Update Mu 
	for (int k = 0; k != fDim; k++)
	{
		tempMu[k] = (GammaX[k] + dDsm * mu[k])/Den;
	}


	//invSigma Update
	double* tempInv = new double[sigmaL()];
	double* tempDiff = new double[sigmaL()];
	double* Sigma = new double[sigmaL()];

	memset(tempInv, 0, sizeof(double) * sigmaL());
	memset(tempDiff, 0, sizeof(double) * sigmaL());
	memcpy(Sigma, invSigma, sizeof(double) * sigmaL());
	MathUtility::inv(Sigma, fDim);

	for(int i = 0; i != MMIElen; i++){
		for (int k = 0; k != fDim; k++)
		{
			for (int l = 0; l <= k; l++)
			{
				tempDiff[k * fDim + l] += (data[i * fDim + k] - tempMu[k]) * 
					(data[i * fDim + l] - tempMu[l]) * mWordGamma[i]; // /m_vDataNum[i]
			}
		}
	}
	for (int k = 0; k != fDim; k++)
	{
		for (int l = 0; l <= k; l++)
		{
			tempInv[k * fDim + l] += tempDiff[k * fDim + l] + dDsm * Sigma[k * fDim + l] + dDsm * (mu[k] - tempMu[k]) * (mu[l] - tempMu[l]);
		}
	}

	for (int i = 0; i < sigmaL(); i++)
	{
		tempInv[i] /= Den;
	}

	for (int k = 0; k != fDim; k++)
	{
		for (int l = k + 1; l != fDim; l++)
		{
			tempInv[k * fDim + l] = tempInv[l * fDim + k];
		}
	}

	bool flag = true;


	{
		double matInv = MathUtility::det(tempInv, fDim);
		if (matInv < 0)
		{
			double t = MathUtility::det(tempDiff, fDim) / pow( Den,fDim);
			double det2 = MathUtility::det(Sigma, fDim);
			printf("matDiff minus!det[%.2e], DiffMat[%.2e] lat det[%.2e]\n", matInv, t , det2);
			m_dDsm = dDsm *2;
			if(m_dDsm<10e-4)
				m_dDsm = 2;
			singleGaussianMMIEupdate();
			flag = false;
		}
		MathUtility::inv(tempInv, fDim);
	}

	if (flag)
	{
		memcpy(mu, tempMu, sizeof(double) * fDim);
		memcpy(invSigma , tempInv, sizeof(double) * sigmaL());
	}

	if (MathUtility::rcond(tempInv, fDim)< 1e-10)
	{
		flag = false;
	}
	else
	{
		flag = true;
	}

	delete []GammaX;
	delete []tempInv;
	delete []Sigma;
	delete []tempMu;
	delete []tempDiff;
	return flag;
}