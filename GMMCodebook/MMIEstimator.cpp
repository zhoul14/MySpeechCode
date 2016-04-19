#include "../CommonLib/Math/MathUtility.h"
#include <iostream>
#include <vector>
#include "../CommonLib/CudaCalc/CUGaussLh.h"
#include "../CommonLib/CudaCalc/CUDiagGaussLh.h"
#include "../CommonLib/CudaCalc/CUShareCovLh.h"
#include <string>
#include <windows.h>
#include "GMMCodebookSet.h"
#include "MMIEstimator.h"
#include "../CommonLib//Math/CMatrixInv.h"
#include <stdlib.h> 
CMMIEstimator::CMMIEstimator(int fDim, int mixNum, int cbType, int maxIter, bool useCuda, double coef, int btNum): GMMEstimator(fDim, mixNum, cbType, maxIter, useCuda, coef, btNum)
{

}
void CMMIEstimator::initMMIE(int ConMatLen, double Dsm)
{
	MMIElen = ConMatLen;
	m_dDsm = Dsm;
	m_vNormc = new double* [ConMatLen];
	m_vDataList = new double*[ConMatLen];
	m_vPrecalcLh = new double*[ConMatLen];
	m_vTgamma = new double*[ConMatLen];
	m_pMMIEgamma = new double[ConMatLen];
	m_pDataMatSelfIdx = new int[ConMatLen];
}


void CMMIEstimator::initMMIECbs(int DataMatLen)
{
	MMIEDataLen = DataMatLen;
	m_vMuList = new double*[DataMatLen];
	m_vAlphaList = new double*[DataMatLen];
	m_vInvList = new double*[DataMatLen];
}

void CMMIEstimator::loadMMIEData(double* x, int dataNum, int idx, int dataId)
{
	m_vDataId.push_back(dataId);
	m_vDataNum.push_back(dataNum); 

	if (idx == m_nSelfProc)
	{
		this->dataNum = dataNum;
	}
	m_vDataList[idx]= new double[dataNum * fDim];
	memcpy(m_vDataList[idx], x, dataNum * fDim * sizeof(double));
	m_vPrecalcLh[idx] = new double[dataNum * mixNum * DataStateMat[dataId].size()];
	memset(m_vPrecalcLh[idx], 0, dataNum * mixNum * sizeof(double) * DataStateMat[dataId].size());
	m_vTgamma[idx] = new double[dataNum];
	memset(m_vTgamma[idx], 0, sizeof(double) * dataNum * mixNum);
}

void CMMIEstimator::loadMMIECbs(double* a, double* m, double * invSgm, int idx)
{
	m_vMuList[idx] = new double[fDim * mixNum];
	m_vAlphaList[idx] = new double[mixNum];
	m_vInvList[idx] = new double[mixNum * sigmaL()];
	memcpy(m_vInvList[idx], invSgm, sigmaL() * mixNum * sizeof(double));
	memcpy(m_vAlphaList[idx], a, mixNum * sizeof(double));
	memcpy(m_vMuList[idx], m, mixNum * fDim * sizeof(double));
}

CMMIEstimator::~CMMIEstimator()
{
}

void CMMIEstimator::setGamma1(int k)
{
	m_pMMIEgamma[k] = 0;

}

void CMMIEstimator::destoyMMIEr()
{
	//destructMat(m_vNormc);
	m_vDataId.clear();
	m_vDataNum.clear();
	destructMat(m_vPrecalcLh, MMIElen);
	destructMat(m_vDataList, MMIElen);
	destructMat(m_vTgamma, MMIElen);
	delete[]m_vNormc;
	delete[]m_vDataList;
	delete[]m_vPrecalcLh; 
	delete[]m_pMMIEgamma;
	delete[]m_pDataMatSelfIdx;
}
void CMMIEstimator::destroyMMIECbs()
{
	destructMat(m_vAlphaList, MMIEDataLen);
	destructMat(m_vMuList, MMIEDataLen);
	destructMat(m_vInvList, MMIEDataLen);
	delete[]m_vInvList;
	delete[]m_vAlphaList;
	delete[]m_vMuList;
}
bool CMMIEstimator::destructMat(void* pMat, int len)
{

	if (pMat == nullptr)
	{
		return false;
	}
	double ** t = (double **)pMat; 

	for (int i = 0; i < len; i++)
	{
		delete [] t[i];
	}
	return true;
}

bool CMMIEstimator::prepareByMMIELh(int CBidx, int fnum){
	if (m_vMMIELHgen == NULL || m_vMMIELH == NULL)
	{
		return false;
	}

	{
		for (int j = 0; j != fnum; j++)
		{
			m_vTgamma;
		}
	}

	m_vMMIELH[CBidx];
	return true;//没写完。

}


double CMMIEstimator::prepareGammaIForUpdate(int procIdx, bool bFirst) 
{

	int selfIdx = -1;
	int dataId = m_vDataId[procIdx];
	int tempDataNum = m_vDataNum[procIdx];
	int conStateNum = DataStateMat[dataId].size();

	if (MMIEDataLen != conStateNum)
	{
		printf("error! multi cbs");
		exit(-1);
	}
	m_pMMIEgamma[procIdx] = 0;


	for (int i = 0; i != conStateNum; i++)
	{
		if (DataStateMat[dataId][i] - 1 == cbId)
		{
			m_pDataMatSelfIdx[procIdx] = i;
			selfIdx = i;
		}
		double* pInvSigma = i == selfIdx ? invSigma: m_vInvList[i];
		double* pMu = i == selfIdx ? mu : m_vMuList[i];

		if ( bFirst || i == selfIdx)
		{
			if (cbType == CB_TYPE_FULL_RANK) {
				CUGaussLh* cugl = new CUGaussLh(mixNum, fDim, pInvSigma, pMu, NULL, -1, useCuda, m_dCoef);
				cugl->runCalc(m_vDataList[procIdx], tempDataNum, m_vPrecalcLh[procIdx] + tempDataNum * mixNum * i);
				delete cugl;
			} else if (cbType == CB_TYPE_DIAG) {
				CUDiagGaussLh* cugl = new CUDiagGaussLh(mixNum, fDim, pInvSigma, pMu, NULL, -1, useCuda);
				cugl->runCalc(m_vDataList[procIdx], tempDataNum, m_vPrecalcLh[procIdx] + tempDataNum * mixNum * i);
				delete cugl;
			} else if (CB_TYPE_FULL_RANK_SHARE) {
				CUShareCovLh* cugl = new CUShareCovLh(mixNum, fDim, mixNum, pInvSigma, pMu, NULL, useCuda);
				cugl->runCalc(m_vDataList[procIdx], tempDataNum, m_vPrecalcLh[procIdx] + tempDataNum * mixNum * i);
				delete cugl;
			}
		}
	}

	double con = (procIdx == m_nSelfProc ? 1 : 0);
	double* temp = new double[conStateNum * tempDataNum];
	for (int i = 0; i < conStateNum; i++)
	{
		for (int j = 0; j < tempDataNum; j++)
		{
			temp[i + j * conStateNum] = MathUtility::logSumExp(m_vPrecalcLh[procIdx] + i * tempDataNum * mixNum + j * mixNum, m_vAlphaList[i], mixNum);
		}
	}
	for (int j = 0; j != tempDataNum; j++)
	{

		double PtAll = MathUtility::logSumExp(temp + j * conStateNum, conStateNum);
		double val = (temp[j * conStateNum + selfIdx] - PtAll);
		val = abs(val) > 307 ? 0 : exp(val);

		for (int m = 0; m != mixNum; m++)
		{
			double valm = m_vPrecalcLh[procIdx][selfIdx * tempDataNum * mixNum + j * mixNum + m] - temp[selfIdx + j * conStateNum];
			valm = abs(valm) > 307 ? 0 :exp(valm) * m_vAlphaList[selfIdx][m];
			m_vTgamma[procIdx][j * mixNum + m] = (con - val) * valm;
			m_pMMIEgamma[procIdx] += (con - val) * valm;
		}		
	}

	// 
	// 	double sumLh = 0.0;
	// 	double idxLh = 0.0;
	// 	double* temp2 = new double[conStateNum * tempDataNum];
	// 	memset(temp2,0,sizeof(double) * conStateNum * tempDataNum);
	// 	for (int i = 0; i < conStateNum; i++)
	// 	{
	// 		for (int j = 0; j < tempDataNum; j++)
	// 		{
	// 			{
	// 				temp2[i] += MathUtility::logSumExp(m_vPrecalcLh[procIdx] + i * tempDataNum * mixNum + j * mixNum, m_vAlphaList[i], mixNum);
	// 			}
	// 		}
	// 		temp2[i] /= tempDataNum;
	// 	}
	// 	double sum = MathUtility::logSumExp(temp2, conStateNum);
	// 	double val = temp2[selfIdx] - sum;
	// 	delete []temp2;
	// 
	// 	m_pMMIEgamma[procIdx] = abs(val) > 308 ? 0 : exp(val);
	// 	m_pMMIEgamma[procIdx] = ((procIdx == m_nSelfProc ? 1 : 0) - m_pMMIEgamma[procIdx] );
	// 	if (procIdx == m_nSelfProc)
	// 	{
	// 		m_pMMIEgamma[procIdx] *= 10;
	// 	}
	// 	/*if (abs(m_pMMIEgamma[procIdx]) > 0.12)
	// 	{
	// 	printf("当前第%d号gamma，值为[%lf],selfId为[%d],val:[%lf]\n",procIdx,m_pMMIEgamma[procIdx],m_nSelfProc,val);
	// 	}*/
	// 	if (procIdx == m_nSelfProc)
	// 	{
	// 		m_pMMIEgamma[procIdx] *= 10;
	// 	}
	m_pMMIEgamma[procIdx];
	delete[]temp;
	if (_isnan(m_pMMIEgamma[procIdx])||!_finite(m_pMMIEgamma[procIdx]))
	{
		printf(" nan procIdx %d",cbId);

	}
	return m_pMMIEgamma[procIdx];
}


bool CMMIEstimator::checkGamma(double&res)
{
	int cnt = 0;
	double sum = 0.0;


	for (int i = 0; i != MMIElen; i++)
	{
		if (m_pMMIEgamma[i] == 0) 
		{
			cnt++;
		}
		sum += m_pMMIEgamma[i];
	}
	//res = (sum) - 2 * m_pMMIEgamma[m_nSelfProc];
	res = m_pMMIEgamma[m_nSelfProc];
// 	if (cnt < MMIElen - 1)
// 	{
// 		return sum == 0;
// 	}
	
	//printf("MMIELEN:[%d],CNT:[%d],SUM:%lf\n",MMIElen,cnt,sum);
	//printf("gamma is norm and need not update\n");
	return m_nSelfProc == -1;	
}


bool CMMIEstimator::singleGaussianMMIEupdate()
{
	if (mixNum != 1) {
		printf("mixNum(%d) != 1, single gaussian update shouldn't be called\n", mixNum);
		exit(-1);
	}

	//precalcMMIEgamma();

	// Mu update
	double* GammaX = new double[fDim];
	double* SumX = new double[fDim];
	double* tempMu = new double[fDim];

	double GammaOne(0.0);
	memset(SumX, 0, fDim * sizeof(double));
	memset(GammaX, 0, fDim * sizeof(double));

	for (int i = 0 ; i != MMIElen; i++)
	{
		for (int j = 0; j != m_vDataNum[i]; j++)
		{
			for (int k = 0; k != fDim; k++)
			{
				//SumX[k] += m_vDataList[i][j * fDim + k]/ m_vDataNum[i];
				GammaX[k] += m_vDataList[i][j * fDim + k] * m_vTgamma[i][j];
			}
			GammaOne += m_vTgamma[i][j];
		}
		// 		for (int k = 0; k != fDim; k++)
		// 		{
		// 			GammaX[k] += m_pMMIEgamma[i] * SumX[k];
		// 		}
		// 		GammaOne += m_pMMIEgamma[i];
	}

	double dDsm = 0;
	for(int i = 0; i != MMIElen; i++)
		for (int j = 0 ; j != m_vDataNum[i]; j++)
		{
			dDsm += abs(i == m_nSelfProc? 1- m_vTgamma[i][j]:m_vTgamma[i][j]) ;
		}
		dDsm = dDsm * 3;
		dDsm = max(dDsm, m_dDsm);
		double Den = GammaOne + dDsm;
		for (int k = 0; k != fDim; k++)
		{
			tempMu[k] = (GammaX[k] + dDsm * mu[k])/Den;
		}

		//invSigma Update
		// (X - MU)(X - MU)T
		double* tempInv = new double[sigmaL()];
		double* tempDiff = new double[sigmaL()];
		double* Sigma = new double[sigmaL()];

		memset(tempInv, 0, sizeof(double) * sigmaL());
		memset(tempDiff, 0, sizeof(double) * sigmaL());
		memcpy(Sigma, invSigma, sizeof(double) * sigmaL());
		MathUtility::inv(Sigma, fDim);

		for (int i = 0 ; i != MMIElen; i++)
		{
			//memset(XM, 0, sizeof(double) * sigmaL());

			for (int j = 0; j != m_vDataNum[i]; j++)
			{

				for (int k = 0; k != fDim; k++)
				{
					for (int l = 0; l <= k; l++)
					{
						tempDiff[k * fDim + l] += (m_vDataList[i][j * fDim + k] - tempMu[k]) * 
							(m_vDataList[i][j * fDim + l] - tempMu[l]) * m_vTgamma[i][j]; // /m_vDataNum[i]
					}
				}
			}
			// 
			// 		for (int k = 0; k != fDim; k++)
			// 		{
			// 			for (int l = 0; l <= k; l++)
			// 			{
			// 				tempDiff[k * fDim + l] += XM[k * fDim + l] * m_pMMIEgamma[i];
			// 			}
			// 		}
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
				printfGamma();

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
		delete []SumX;
		delete []tempInv;
		delete []Sigma;
		delete []tempMu;
		delete []tempDiff;
		return flag;
}
bool CMMIEstimator::MixtureGaussianMMIEupdate()
{
	// Mu update
	bool flag = true;

	double* SumX = new double[fDim * mixNum];
	double* tempMu = new double[fDim * mixNum];
	double* dDsm = new double[mixNum];
	double* GammaOne = new double[mixNum];
	double* Den = new double[mixNum];
	memset(SumX, 0, fDim * mixNum *sizeof(double));
	memset(dDsm, 0 , mixNum * sizeof(double));
	memset(GammaOne, 0 , mixNum * sizeof(double));
	memset(Den, 0, mixNum * sizeof(double));

	int* datalen = new int[MMIElen + 1];
	memset(datalen, 0, sizeof(int) * MMIElen + sizeof(int));
	for (int i = 1; i != MMIElen + 1; i++) datalen[i] += m_vDataNum[i - 1] + datalen[i-1];

	double* MMIEgamma = new double[datalen[MMIElen] * mixNum];
	memset(MMIEgamma, 0, sizeof(double) * datalen[MMIElen] * mixNum);

	int idx = 0;
	for (int i = 0; i != MMIElen; i++)//review
	{
		for (int j = 0; j != m_vDataNum[i];j++)
		{
			double gamma = 0;
			int idx2 = m_pDataMatSelfIdx[i] * m_vDataNum[i] * mixNum + j * mixNum;
			gamma -= MathUtility::logSumExp(m_vPrecalcLh[i] +  idx2, alpha, mixNum);
			for (int m = 0; m != mixNum; m++)
			{
				MMIEgamma[idx++]=exp(gamma + m_vPrecalcLh[i][idx2 + m]) * alpha[m] * m_pMMIEgamma[i];
			}
		}
	}

	for (int m = 0; m != mixNum; m++)
	{
		for (int i = 0 ; i != MMIElen; i++)
		{
			for (int j = 0; j != m_vDataNum[i]; j++)
			{
				idx = datalen[i] * mixNum + j * mixNum + m;
				for (int k = 0; k != fDim; k++)
				{
					SumX[k + m * fDim] += m_vDataList[i][j * fDim + k] / m_vDataNum[i] * MMIEgamma[idx];
				}
				GammaOne[m] += MMIEgamma[idx] / m_vDataNum[i];
			}
		}

		for(int i = 0; i != MMIElen; i++)
		{
			dDsm[m] += abs(m_pMMIEgamma[i]) ;
		}
		dDsm[m] = dDsm[m]*m_dDsm;
		Den[m] = GammaOne[m] + dDsm[m];

		for (int k = 0; k != fDim; k++)
		{
			tempMu[k + m * fDim] = (SumX[k] + dDsm[m] * mu[k + m * fDim])/Den[m];
		}
	}
	//invSigma Update
	// (X - MU)(X - MU)T
	double* XM = new double [mixNum * sigmaL()];
	double* tempInv = new double[mixNum * sigmaL()];
	double* tempDiff = new double[mixNum * sigmaL()];
	double* Sigma = new double[mixNum * sigmaL()];

	memset(tempInv, 0, sizeof(double) * mixNum * sigmaL());
	memset(tempDiff, 0, sizeof(double) * mixNum * sigmaL());
	memcpy(Sigma, invSigma, sizeof(double) * mixNum * sigmaL());
	memset(XM, 0, sizeof(double) * sigmaL() * mixNum);


	for (int m = 0; m != mixNum; m++)
	{
		MathUtility::inv(Sigma + m * sigmaL(), fDim);

		for (int i = 0 ; i != MMIElen; i++)
		{

			for (int j = 0; j != m_vDataNum[i]; j++)
			{
				idx = datalen[i] * mixNum + j * mixNum + m;
				for (int k = 0; k != fDim; k++)
				{
					for (int l = 0; l <= k; l++)
					{
						XM[k * fDim + l + m * sigmaL()] += (m_vDataList[i][j * fDim + k] - tempMu[k + m *fDim]) * MMIEgamma[idx] *
							(m_vDataList[i][j * fDim + l] - tempMu[l + m *fDim]) / m_vDataNum[i];// /m_vDataNum[i]
					}
				}
			}

		}

		for (int k = 0; k != fDim; k++)
		{
			for (int l = 0; l <= k; l++)
			{
				tempInv[k * fDim + l + m * sigmaL()] += XM[k * fDim + l] + dDsm[m] * Sigma[k * fDim + l + m * sigmaL()] + dDsm[m] * (mu[k + m * fDim] - tempMu[k + m *fDim]) * (mu[l + m * fDim] - tempMu[l + m *fDim]);
			}
		}
		for (int i = 0; i < sigmaL(); i++)
		{
			tempInv[i + m * sigmaL()] /= Den[m];
		}

		for (int k = 0; k != fDim; k++)
		{
			for (int l = k + 1; l != fDim; l++)
			{
				tempInv[k * fDim + l + m * sigmaL()] = tempInv[l * fDim + k + m * sigmaL()];
			}
		}

		double matInv = MathUtility::det(tempInv + m * sigmaL(), fDim);
		if (matInv < 0)
		{
			double t = MathUtility::det(tempDiff + m * sigmaL(), fDim) / pow( Den[m],fDim);
			double det2 = MathUtility::det(Sigma + m * sigmaL(), fDim);
			printf("matDiff minus!det[%.2e], DiffMat[%.2e] lat det[%.2e]\n", matInv, t , det2);
			printfGamma();
			flag = false;
		}

		MathUtility::inv(tempInv + m * sigmaL(), fDim);
		if (MathUtility::rcond(tempInv + m * sigmaL(), fDim)< 1e-10)
		{
			flag = false;
		}

	}

	if (flag)
	{
		memcpy(mu, tempMu, sizeof(double) * fDim * mixNum);
		memcpy(invSigma , tempInv, sizeof(double) * sigmaL() * mixNum);
	}
	//alpha update
	//review
	m_pMMIEgamma[m_nSelfProc] = m_pMMIEgamma[m_nSelfProc]-1;

	double gamma_spk_sum = 0.0;
	double* gamma_spk = new double[mixNum];
	memset(gamma_spk, 0, sizeof(double) * mixNum);
	double * pPrelh = m_vPrecalcLh[m_nSelfProc] +  m_pDataMatSelfIdx[m_nSelfProc] * dataNum * mixNum;
	for (int m = 0; m != mixNum; m++)
	{
		for (int j = 0; j != dataNum; j++)
		{
			double genlh = MathUtility::logSumExp(pPrelh + j * mixNum + m, alpha, mixNum);
			gamma_spk[m] += alpha[m] * exp(pPrelh[j * mixNum + m] - genlh);
		}
		gamma_spk_sum += gamma_spk[m];
	}

	double gamma_gen_sum = 0.0;
	double* gamma_gen = new double[mixNum];
	memset(gamma_gen, 0, sizeof(double) * mixNum);
	for (int m = 0; m!= mixNum; m++)
	{
		for (int r = 0; r != MMIElen; r++)
		{
			for (int j = 0; j != m_vDataNum[r]; j++)
			{
				int i =  m_pDataMatSelfIdx[r] * m_vDataNum[r] * mixNum;
				double sumLh = MathUtility::logSumExp(m_vPrecalcLh[r]+ i + j * mixNum, alpha, mixNum);
				gamma_gen[m] += exp(m_vPrecalcLh[r][i + j * mixNum + m] - sumLh) * alpha[m];
			}
			gamma_gen[m] = m_pMMIEgamma[r] * gamma_gen[m] / m_vDataNum[r];
		}
		gamma_gen_sum += gamma_gen[m];
	}

	double* gammadiff = new double[mixNum];
	double gammadiffsum = 0.0;
	double cs = 5;

	for (int m = 0; m!= mixNum; m++)
	{
		gammadiff[m] = gamma_spk[m]/gamma_spk_sum - gamma_gen[m]/gamma_gen_sum;
		gammadiffsum += gammadiff[m]*alpha[m];
	}
	for (int m = 0; m!= mixNum; m++)
	{
		alpha[m] *= (gammadiff[m] + cs)/(gammadiffsum + cs);
	}


	delete []gammadiff;
	delete []gamma_gen;
	delete []gamma_spk;
	delete []SumX;
	delete []tempInv;
	delete []XM;
	delete []Sigma;
	delete []tempMu;
	delete []tempDiff;
	delete []MMIEgamma;
	delete []datalen;
	delete []GammaOne;
	delete []Den;
	delete []dDsm;
	return flag;
}


int CMMIEstimator::estimate()
{
	if (dataNum < fDim * mixNum / 2) {
		return SAMPLE_NOT_ENOUGH;
	}
	int res = 0;
	for (int i = 0; i < 1/*maxIter*/; i++)
	{
		if (mixNum == 1) 
		{
			m_dDsm = -2;
			res = singleGaussianMMIEupdate() ? SUCCESS : ILL_CONDITIONED;
		}
		else
		{
			res = MixtureGaussianMMIEupdate() ? SUCCESS : ILL_CONDITIONED;
		}
	}
	return res;
}

void CMMIEstimator::printfGamma(FILE* logFile)
{
	fprintf(logFile,"gamma No:[%d], value[%lf]\t",m_nSelfProc,m_pMMIEgamma[m_nSelfProc]);

	for (int i =0; i < MMIElen; i++)
	{
		if (i != m_nSelfProc)
		{
			//continue;
		}
		fprintf(logFile,"gamma No:[%d], value[%lf]\t",i,m_pMMIEgamma[i]);
	}
	fprintf(logFile,"\nMDSM = {%lf}\n",m_dDsm);
}

