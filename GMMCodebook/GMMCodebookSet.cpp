#include "GMMCodebookSet.h"
#include "../CommonLib/Math/MathUtility.h"
#include <cassert>
#include "iostream"
GMMCodebook GMMCodebookSet::getCodebook(int num)
{
	if (num >= CodebookNum)
	{
		printf("num not in range: 0 - %d\n", CodebookNum - 1);
		exit(-1);
		return GMMCodebook(0, 0, cbType);
	}
	else
	{
		int L = SigmaL();
		GMMCodebook cb(MixNum, FDim, cbType);
		memcpy(cb.Mu, Mu + num * MixNum * FDim, MixNum * FDim * sizeof(double));
		memcpy(cb.InvSigma, InvSigma + num * L, L * sizeof(double));
		memcpy(cb.Alpha, Alpha + num * MixNum, MixNum * sizeof(double));
		cb.DurMean = DurMean[num];
		cb.DurVar = DurVar[num];
		cb.DETSIGMA = DetSigma[num];
		if (isFullMix())
		{
			memcpy(cb.Beta, Beta + num * MixNum * BetaNum, BetaNum * MixNum * sizeof(double));
		}
		cb.BetaNum = BetaNum;
		return cb;

	}
}

void GMMCodebookSet::printCodebookSetToFile(const char* filename)
{
	printf("Printing CodebookSet\n");
	FILE* fid = fopen(filename, "w");
	for (int i = 0; i < CodebookNum; i++)
	{
		GMMCodebook c (getCodebook(i));

		fprintf(fid, "GMMCODEBOOK_NO_%d", i);
		if (c.cbType == CB_TYPE_FULL_RANK)
			fprintf(fid, " FULL\n");
		else if (c.cbType == CB_TYPE_DIAG)
			fprintf(fid, " DIAG\n");
		else if (c.cbType == CB_TYPE_FULL_RANK_SHARE)
			fprintf(fid, " FULL_SHARED\n");
		else if (c.cbType == CB_TYPE_FULL_MIX)
			fprintf(fid, " FULL_MIX\n");

		fprintf(fid, "DURMEAN\n%f\n", c.DurMean);
		fprintf(fid, "DURSIGMA\n%f\n", sqrt(c.DurVar));
		fprintf(fid, "DETSIGMA\n\%f\n", c.DETSIGMA);

		fprintf(fid, "ALPHA\n");
		for (int j = 0; j < MixNum; j++)
			fprintf(fid, "%f\t", c.Alpha[j]);
		fprintf(fid, "\n");

		for (int j = 0; j < MixNum; j++)
		{
			fprintf(fid, "MU_MIX_%d\n", j);
			for (int k = 0; k < FDim; k++)
				fprintf(fid, "%f\t", c.Mu[j * FDim + k]);
			fprintf(fid, "\n");
		}

		if (c.cbType == CB_TYPE_FULL_MIX)
		{
			fprintf(fid, "BETA_%d\n",BetaNum);
			for (int j = 0; j < BetaNum;j++)
			{
				fprintf(fid,"%f\t", c.Beta[j]);
			}
			fprintf(fid,"\n");
		}

		if (c.cbType != CB_TYPE_FULL_RANK_SHARE) {
			for (int j = 0; j < MixNum; j++) {
				fprintf(fid, "INV_SIGMA_MIX_%d\n", j);
				if (c.cbType == CB_TYPE_FULL_RANK) {
					for (int k = 0; k < FDim; k++) {
						for (int l = 0; l < FDim; l++)
							fprintf(fid, "%f\t", c.InvSigma[j * FDim * FDim + k * FDim + l]);
						fprintf(fid, "\n");
					}
				} else if (c.cbType == CB_TYPE_DIAG) {
					for (int k = 0; k < FDim; k++)
						fprintf(fid, "%f\t", c.InvSigma[j * FDim + k]);
					fprintf(fid, "\n");
				} else if(c.cbType == CB_TYPE_FULL_MIX){
					for (int k = 0; k < FEATURE_DIM; k++)
					{
						for (int l = 0; l < FEATURE_DIM; l++)
						{
							fprintf(fid, "%f\t", c.InvSigma[j * c.SigmaL() + k * FEATURE_DIM + l]);
						}
					}
					fprintf(fid, "\n");
					for (int k =  0; k < FDim - FEATURE_DIM; k++)
					{
						for (int l = 0; l < FDim - FEATURE_DIM; l++)
						{
							fprintf(fid, "%f\t", c.InvSigma[j * c.SigmaL() + k * (FDim - FEATURE_DIM) + l + FEATURE_DIM * FEATURE_DIM]);
						}
					}
					fprintf(fid, "\n");

				}


			}
		} else {
			fprintf(fid, "INV_SIGMA_SHARED\n");
			for (int j = 0; j < FDim; j++) {
				for (int k = 0; k < FDim; k++) {
					fprintf(fid, "%f\t", c.InvSigma[j * FDim + k]);
				}
				fprintf(fid, "\n");
			}
		}

		fprintf(fid, "\n");

	}
	fclose(fid);
	return;
}


GMMCodebookSet::GMMCodebookSet(const char* initBuf, int mode,double coef) {
	if (mode == INIT_MODE_FILE) {
		initializeFromOutside = false;
		FILE* fid = fopen(initBuf, "rb");
		if (!fid) {
			printf("cannot open gmm codebook file[%s]", initBuf);;
			exit(-1);
		}

		fread(&CodebookNum, sizeof(int), 1, fid);
		fread(&FDim, sizeof(int), 1, fid);
		fread(&MixNum, sizeof(int), 1, fid);
		fread(&cbType, sizeof(int), 1, fid);
		m_Coef = coef;

		if (cbType < CB_TYPE_FULL_MIX)
		{
			int L = SigmaL();
			int M = isSharedSigma() ? 1 : MixNum;
			Mu = new double[CodebookNum * MixNum * FDim];
			InvSigma = new double[CodebookNum * L];
			Alpha = new double[CodebookNum * MixNum];
			DurMean = new double[CodebookNum];
			DurVar = new double[CodebookNum];
			LogDurVar = new double[CodebookNum];
			DetSigma = new double[CodebookNum * M];

			Beta = NULL;
			BetaNum = 0;

			fread(Alpha, sizeof(double), MixNum * CodebookNum, fid);
			fread(Mu, sizeof(double), MixNum * CodebookNum * FDim, fid);
			fread(InvSigma, sizeof(double), CodebookNum * L, fid);
			fread(DurMean, sizeof(double), CodebookNum, fid);
			fread(DurVar, sizeof(double), CodebookNum, fid);
			fread(LogDurVar, sizeof(double), CodebookNum, fid);
			fread(DetSigma, sizeof(double), CodebookNum * M, fid);
		}
		else
		{
			int L = SigmaL();
			int M = isSharedSigma() ? 1 : MixNum;
			fread(&BetaNum, sizeof(int), 1, fid);
			Mu = new double[CodebookNum * MixNum * FDim];
			InvSigma = new double[CodebookNum * L];
			Alpha = new double[CodebookNum * MixNum];
			DurMean = new double[CodebookNum];
			DurVar = new double[CodebookNum];
			LogDurVar = new double[CodebookNum];
			DetSigma = new double[CodebookNum * M * BetaNum];
			Beta = new double[MixNum * CodebookNum * BetaNum];
			fread(Alpha, sizeof(double), MixNum * CodebookNum, fid);
			fread(Beta, sizeof(double), MixNum * BetaNum * CodebookNum, fid);
			fread(Mu, sizeof(double), MixNum * CodebookNum * FDim, fid);
			fread(InvSigma, sizeof(double), CodebookNum * L, fid);
			fread(DurMean, sizeof(double), CodebookNum, fid);
			fread(DurVar, sizeof(double), CodebookNum, fid);
			fread(LogDurVar, sizeof(double), CodebookNum, fid);
			fread(DetSigma, sizeof(double), CodebookNum * M *BetaNum, fid);
		}
		//fread(&isFullRank, sizeof(bool), 1, fid);

		fclose(fid);
	} else if (mode == INIT_MODE_MEMORY) {
		initializeFromOutside = true;
		const char* p = initBuf;
		memcpy(&CodebookNum, p, sizeof(int));
		p += sizeof(int);
		memcpy(&FDim, p, sizeof(int));
		p += sizeof(int);
		memcpy(&MixNum, p, sizeof(int));
		p += sizeof(int);
		memcpy(&cbType, p, sizeof(int));
		p += sizeof(int);


		int L = SigmaL();
		Alpha = (double*)p;
		Mu = Alpha + MixNum * CodebookNum;
		InvSigma = Mu + MixNum * CodebookNum * FDim;
		DurMean = InvSigma + CodebookNum * L;
		DurVar = DurMean + CodebookNum;
		LogDurVar = DurVar + CodebookNum;
		DetSigma = LogDurVar + CodebookNum;

	}

}

void GMMCodebookSet::updateCodebook(int cbIdx, GMMCodebook& cb) {
	if (cb.cbType != cbType) {
		printf("Error in updateCodebook: cb's fullRankFlag not equal to codebookset's fullRankFlag\n");
		exit(-1);
	}
	updateCodebook(&cbIdx, 1, cb.Alpha, cb.Mu, cb.InvSigma, &cb.DurMean, &cb.DurVar, cb.Beta);
}

void GMMCodebookSet::updateCodebook(int* Idx, int updateNum, double* alpha, double* mu, double* invsigma, double* durmean, double* durvar,double * beta)
{
	if (alpha != NULL)
	{
		for (int i = 0; i < updateNum; i++)
			memcpy(Alpha + MixNum * Idx[i], alpha + MixNum * i, MixNum * sizeof(double));
	}
	if (mu != NULL)
	{
		for (int i = 0; i < updateNum; i++)
			memcpy(Mu + MixNum * FDim * Idx[i], mu + MixNum * FDim * i, MixNum * FDim * sizeof(double));
	}

	if (invsigma != NULL)
	{
		int L = SigmaL();
		for (int i = 0; i < updateNum; i++)
			memcpy(InvSigma + L * Idx[i], invsigma + i * L, L * sizeof(double));
	}

	if (durmean != NULL)
	{
		for (int i = 0; i < updateNum; i++)
			memcpy(DurMean + Idx[i], durmean + i, sizeof(double));
	}

	if (durvar != NULL)
	{
		for (int i = 0; i < updateNum; i++)
			memcpy(DurVar + Idx[i], durvar + i, sizeof(double));
		for (int i = 0; i < updateNum; i++)
			LogDurVar[Idx[i]] = log(durvar[i]);
	}

	if (beta != NULL)
	{
		for(int i = 0; i <updateNum; i++)
			memcpy(Beta + Idx[i] * BetaNum *MixNum, beta + BetaNum * MixNum * i, sizeof(double)* BetaNum * MixNum);
	}

	renewDetSigma(Idx, updateNum);
}

void GMMCodebookSet::renewDetSigma(int* Idx, int updateNum)
{
	bool idxNullFlag = (Idx == NULL);
	if (idxNullFlag) {
		Idx = new int[CodebookNum];
		for (int i = 0; i < CodebookNum; i++) {
			Idx[i] = i;
		}
		updateNum = CodebookNum;
	}

	int difDim = FDim - g_DefaultFdim >0? FDim - g_DefaultFdim: 0;
	

	double *pAppendDet = new double[difDim*difDim];

	if (!isSharedSigma()) {
		for (int i = 0; i < updateNum; i++)
		{
			for (int j = 0; j < MixNum; j++)
			{
				int p = Idx[i] * MixNum + j;
				if (isFullRank()){
					DetSigma[p] = 1 / MathUtility::det(InvSigma + p * FDim * FDim, FDim);
					if (difDim && m_Coef > 1e-6)
					{
						DetSigma[p] *= pow(m_Coef, difDim);				
					}
					else if (m_Coef < 1e-6 && difDim)
					{
						for (int k = 0; k < FDim - g_DefaultFdim; k ++)
						{
							memcpy(pAppendDet + k * difDim, InvSigma + p *FDim * FDim + FDim * g_DefaultFdim + g_DefaultFdim + k * FDim, sizeof(double) * difDim);
						}
						DetSigma[p] *= MathUtility::det(pAppendDet,difDim);
					}
				}
				else if (isFullMix())
				{
					DetSigma[2 * p] = 1 / MathUtility::det(InvSigma + p * SigmaL()/MixNum , FEATURE_DIM);
					DetSigma[2 * p + 1] = 1 / MathUtility::det(InvSigma + p * SigmaL()/MixNum + FEATURE_DIM * FEATURE_DIM, FDim - FEATURE_DIM);
				}
				else{
					DetSigma[p] = 1 / MathUtility::prod(InvSigma + p * FDim, FDim);
				}
			}
		}
	} else {
		for (int i = 0; i < updateNum; i++) {
			int p = Idx[i];

			double* tmat = InvSigma + p * FDim * FDim;

			DetSigma[p] = 1 / MathUtility::det(InvSigma + p * FDim * FDim, FDim);
		}
	}

	delete [] pAppendDet;
	if (idxNullFlag) {
		delete [] Idx;
	}
}


bool GMMCodebookSet::saveCodebook(const std::string& filename)
{
	if (Alpha == NULL || Mu == NULL || InvSigma == NULL || DurMean == NULL ||
		DurVar == NULL || DetSigma == NULL || CodebookNum == 0 || FDim == 0)
	{
		printf("cannot save codebook because some of its elements is null or 0\n");
		exit(-1);
	}

	FILE* fid = fopen(filename.c_str(), "wb");
	std::cout<<(filename);
	if (!fid)
	{
		printf("cannot write gmm codebook\n");
		exit(-1);
	}

	printf("Saving GMM Codebook... CodebookNum = %d, MixNum = %d\r", CodebookNum, MixNum);
	fwrite(&CodebookNum, sizeof(int), 1, fid);
	fwrite(&FDim, sizeof(int), 1, fid);
	fwrite(&MixNum, sizeof(int), 1, fid);
	fwrite(&cbType, sizeof(int), 1, fid);

	int M = isSharedSigma() ? 1 : MixNum;

	if (isFullMix())
	{
		fwrite(&BetaNum, sizeof(int), 1, fid);
	}

	fwrite(Alpha, sizeof(double), MixNum * CodebookNum, fid);

	if (isFullMix())
	{
		fwrite(Beta, sizeof(double), MixNum * BetaNum * CodebookNum, fid);
	}
	fwrite(Mu, sizeof(double), MixNum * CodebookNum * FDim, fid);
	fwrite(InvSigma, sizeof(double), CodebookNum * SigmaL(), fid);
	fwrite(DurMean, sizeof(double), CodebookNum, fid);
	fwrite(DurVar, sizeof(double), CodebookNum, fid);
	fwrite(LogDurVar, sizeof(double), CodebookNum, fid);
	if (isFullMix())
	{
		fwrite(DetSigma, sizeof(double), CodebookNum * M *BetaNum, fid);
	}
	else
	{
		fwrite(DetSigma, sizeof(double), CodebookNum * M , fid);

	}

	fclose(fid);

	printf("saving GMM codebook ends, codebookNum = %d, mixNum = %d\n", CodebookNum, MixNum);
	return true;
}

int GMMCodebookSet::saveCodebookDSP(char * FileName)
{
	FILE* fid  = fopen(FileName,"wb+");
	fwrite(&cbType, sizeof(int), 1, fid);
	fwrite(&MixNum, sizeof(int), 1, fid);
	float* a = new float[MixNum * CodebookNum];
	float* m = new float[MixNum * CodebookNum * FDim];
	float* in = new float[CodebookNum * SigmaL()];
	float* LogDetSigma  = new float[MixNum * CodebookNum];
	for (int i = 0; i < MixNum * CodebookNum; i++)
	{
		float x = 1/MathUtility::prod(InvSigma + SigmaL()/ MixNum * i,  SigmaL() / MixNum);
		LogDetSigma[i] = log(x);
	}

	for (int i = 0; i < MixNum * CodebookNum; i++)
	{
		a[i] = (float)Alpha[i];
		for (int j = 0; j < FDim; j++)
		{
			m[i * FDim + j] = (float)Mu[i * FDim + j];
			in[i * FDim + j] = (float)InvSigma[i * FDim + j];
		}
	}


	fwrite(a, sizeof(float), MixNum * CodebookNum, fid);
	fwrite(m, sizeof(float), MixNum * CodebookNum * FDim, fid);
	fwrite(in, sizeof(float), CodebookNum * SigmaL(), fid);
	fwrite(LogDetSigma, sizeof(float), MixNum * CodebookNum , fid);

	delete []LogDetSigma;
	delete []a;

	fclose(fid);
	return 1;
}

int GMMCodebookSet::getCodebookNum() const
{
	return CodebookNum;
}

int GMMCodebookSet::getFDim() const
{
	return FDim;
}

int GMMCodebookSet::getMixNum() const
{
	return MixNum;
}


GMMCodebookSet::GMMCodebookSet(int cbnum, int fdim, int mixnum, int cbType, double coef)
{
	CodebookNum = cbnum;
	FDim = fdim;
	MixNum = mixnum;
	this->cbType = cbType;
	initializeFromOutside = false;
	m_Coef = coef;
	int L = SigmaL();
	Mu = new double[CodebookNum * MixNum * FDim];
	InvSigma = new double[CodebookNum * L];
	Alpha = new double[CodebookNum * MixNum];
	DurMean = new double[CodebookNum];
	DurVar = new double[CodebookNum];
	LogDurVar = new double[CodebookNum];
	Beta = NULL;
	BetaNum = 0;

	if (cbType == CB_TYPE_FULL_RANK_SHARE)
		DetSigma = new double[CodebookNum];
	else if (cbType != CB_TYPE_FULL_MIX)
	{
		DetSigma = new double[CodebookNum * MixNum];
	}
	else 
	{
		BetaNum = 2;
		Beta = new double [CodebookNum * MixNum *BetaNum];
		DetSigma = new double [CodebookNum * MixNum * BetaNum];
	}

	memset(DurVar, 0, CodebookNum * sizeof(double));
	memset(LogDurVar, 0, CodebookNum * sizeof(double));
	memset(DurMean, 0, CodebookNum * sizeof(double));
	memset(Mu,0,CodebookNum * MixNum * FDim * sizeof(double));
	memset(Alpha,0,CodebookNum*MixNum *  sizeof(double));
	for(int ii=0;ii<CodebookNum;ii++)
	{
		for (int jj = 0; jj < MixNum; jj++)
		{
			Alpha[ii*jj] = 1 / MixNum;
		}
		for (int jj = 0; jj < L; jj++)
		{
			InvSigma[ii*jj] = 1;			
		}
		DetSigma[ii]=1;
		for (int jj = 0; jj < BetaNum * MixNum; jj++)
		{
			if (jj)
			{
				Beta[ii*BetaNum +jj] = (double)1 / BetaNum - 0.2;
			}
			else
				Beta[ii*BetaNum +jj] = (double)1 / BetaNum + 0.2;
		}
	}
	return;
}

int GMMCodebookSet::getCbType() const {
	return cbType;

}

bool GMMCodebookSet::isFullRank() const {
	return cbType != CB_TYPE_DIAG && cbType !=CB_TYPE_FULL_MIX;
}

bool GMMCodebookSet::isFullMix() const{
	return cbType == CB_TYPE_FULL_MIX;
}

GMMCodebookSet::~GMMCodebookSet()
{
	if (!initializeFromOutside) {
		delete [] Mu;
		delete [] InvSigma;
		delete [] Alpha;
		delete [] DurMean;
		delete [] DurVar;
		delete [] LogDurVar;
		delete [] DetSigma;
		if (Beta)
		{
			delete []Beta;
		}
	}

}

void GMMCodebookSet::splitTwoMu(double* invSigmaPtr, double* oldMuPtr, double* newMuPtr1, double* newMuPtr2, double offset) {

	if (oldMuPtr != newMuPtr1)
		memcpy(newMuPtr1, oldMuPtr, FDim * sizeof(double));
	if (oldMuPtr != newMuPtr2)
		memcpy(newMuPtr2, oldMuPtr, FDim * sizeof(double));

	if (isFullRank()) {
		double eigval = 0;
		double* eigvec = new double[FDim];
		MathUtility::smallest_eig_sym(invSigmaPtr, FDim, &eigval, eigvec);
		//double maxSigma = sqrt(1 / eigval);

		for (int k = 0; k < FDim; k++) {
			newMuPtr1[k] += offset * eigvec[k];
		}

		for (int k = 0; k < FDim; k++) {
			newMuPtr2[k] -= offset * eigvec[k];
		}

		delete [] eigvec;
	} else {
		int p = -1;
		double minIS = 0;
		for (int k = 0; k < FDim; k++) {
			if (p < 0 || invSigmaPtr[k] < minIS) {
				p = k;
				minIS = invSigmaPtr[k];
			}
		}
		newMuPtr1[p] += offset;
		newMuPtr2[p] -= offset;
	}
}

void GMMCodebookSet::checkEqual(GMMCodebookSet* other) {
	assert(this->CodebookNum == other->CodebookNum);
	assert(this->FDim == other->FDim);
	assert(this->MixNum == other->MixNum);
	assert(this->cbType == other->cbType);

	for (int i = 0; i < CodebookNum * MixNum; i++) {
		assert(this->Alpha[i] == other->Alpha[i]);
	}
	printf("check Alpha passed\n");

	for (int i = 0; i < CodebookNum * MixNum * FDim; i++) {
		assert(this->Mu[i] == other->Mu[i]);
	}
	printf("check Mu passed\n");

	int L = SigmaL();
	for (int i = 0; i < CodebookNum * L; i++) {
		assert(this->InvSigma[i] == other->InvSigma[i]);
	}
	printf("check InvSigma passed\n");

	for (int i = 0; i < CodebookNum; i++) {
		assert(this->DurMean[i] == other->DurMean[i]);
	}
	printf("check DurMean passed\n");

	for (int i = 0; i < CodebookNum; i++) {
		assert(this->DurVar[i] > 0 && this->DurVar[i] == other->DurVar[i]);
	}
	printf("check DurVar passed\n");

	for (int i = 0; i < CodebookNum; i++) {
		assert(this->LogDurVar[i] == other->LogDurVar[i]);
	}
	printf("check LogDurVar passed\n");

	for (int i = 0; i < CodebookNum * MixNum; i++) {
		assert(this->DetSigma[i] == other->DetSigma[i]);
	}
	printf("check DetSigma passed\n");
}

void GMMCodebookSet::fillZeroAlpha(double offset) {

	int T = isFullRank() ? FDim * FDim : FDim;
	for (int i = 0; i < CodebookNum; i++) {
		double* tmpAlpha = Alpha + i * MixNum;

		int zeroCnt = 0;
		int* zeroIdx = new int[MixNum];
		for (int j = 0; j < MixNum; j++) {
			if (tmpAlpha[j] == 0) {
				zeroIdx[zeroCnt++] = j;
			}
		}

		for (int j = 0; j < zeroCnt; j++) {
			int zIdx = zeroIdx[j];
			double maxAlpha = -1;
			int maxAlphaIdx = -1;
			for (int k = 0; k < MixNum; k++) {
				if (tmpAlpha[k] > maxAlpha) {
					maxAlpha = tmpAlpha[k];
					maxAlphaIdx = k;
				}
			}

			double* maxMu = Mu + i * MixNum * FDim + maxAlphaIdx * FDim;
			double* zeroMu = Mu + i * MixNum * FDim + zIdx * FDim;
			memcpy(zeroMu, maxMu, FDim * sizeof(double));

			tmpAlpha[maxAlphaIdx] /= 2;
			tmpAlpha[zIdx] = tmpAlpha[maxAlphaIdx];

			if (!isSharedSigma()) {
				double* maxInvSigma = InvSigma + i * MixNum * T + maxAlphaIdx * T;
				double* zeroInvSigma = InvSigma + i * MixNum * T + zIdx * T;
				memcpy(zeroInvSigma, maxInvSigma, T * sizeof(double));

				double* tmpDetSigma = DetSigma + i * MixNum;
				tmpDetSigma[zIdx] = tmpDetSigma[maxAlphaIdx];
				splitTwoMu(maxInvSigma, maxMu, maxMu, zeroMu, offset);
			} else {
				splitTwoMu(InvSigma + i * FDim * FDim, maxMu, maxMu, zeroMu, offset);
			}
		}
		delete [] zeroIdx;
	}
}

void GMMCodebookSet::split2(double offset) {

	fillZeroAlpha(offset);

	int oldMixNum = MixNum;
	int newMixNum = MixNum * 2;
	int newL = SigmaL() * 2;
	double* newAlpha = new double[CodebookNum * newMixNum];
	double* newMu = new double[CodebookNum * newMixNum * FDim];
	double* newInvSigma = NULL;
	double* newDetSigma = NULL;
	if (!isSharedSigma()) {
		newInvSigma = new double[CodebookNum * newL];
		newDetSigma = new double[CodebookNum * newMixNum];
	}

	int T = isFullRank() ? FDim * FDim : FDim;
	for (int i = 0; i < CodebookNum * oldMixNum; i++) {
		newAlpha[i * 2] = Alpha[i] / 2;
		newAlpha[i * 2 + 1] = Alpha[i] / 2;
		if (!isSharedSigma()) {
			newDetSigma[i * 2] = DetSigma[i];
			newDetSigma[i * 2 + 1] = DetSigma[i];

			double* oldInvSigmaPtr = InvSigma + i * T;
			double* newInvSigmaPtr1 = newInvSigma + i * 2 * T;
			double* newInvSigmaPtr2 = newInvSigmaPtr1 + T;
			memcpy(newInvSigmaPtr1, oldInvSigmaPtr, T * sizeof(double));
			memcpy(newInvSigmaPtr2, oldInvSigmaPtr, T * sizeof(double));
		}	
	}

	for (int i = 0; i < CodebookNum; i++) {
		for (int j = 0; j < oldMixNum; j++) {
			double* invSigmaPtr = NULL;
			if (isSharedSigma())
				invSigmaPtr = InvSigma + i * T;
			else
				invSigmaPtr = InvSigma + T * oldMixNum * i + T * j;
			double* newMuPtr1 = newMu + FDim * newMixNum * i + FDim * j * 2;
			double* newMuPtr2 = newMuPtr1 + FDim;
			double* oldMuPtr = Mu + FDim * oldMixNum * i + FDim * j;
			splitTwoMu(invSigmaPtr, oldMuPtr, newMuPtr1, newMuPtr2, offset);

		}
	}

	delete [] Alpha;
	Alpha = newAlpha;
	delete [] Mu;
	Mu = newMu;

	if (!isSharedSigma()) {
		delete [] InvSigma;
		InvSigma = newInvSigma;

		delete [] DetSigma;
		DetSigma = newDetSigma;

	}

	MixNum = newMixNum;
}

void GMMCodebookSet::myBubbleSort(double* val, int n, int* idx) {


	bool* detected = new bool[n];
	for (int i = 0; i < n; i++) 
		detected[i] = false;

	for (int i = 0; i < n; i++) {
		int maxIdx = -1;
		double maxVal = 0;
		for (int j = 0; j < n; j++) {
			if (detected[j])
				continue;

			if (maxIdx == -1 || val[j] > maxVal) {
				maxIdx = j;
				maxVal = val[j];
			}
		}
		idx[i] = maxIdx;
		detected[maxIdx] = true;
	}
	delete [] detected;
}

void GMMCodebookSet::splitAddN(int addN, double offset) {
	if (addN > MixNum) {
		printf("splitAddN: n[%d] should less than MixNum[%d]\n", addN, MixNum);
		exit(-1);
	}
	fillZeroAlpha(offset);

	int oldMixNum = MixNum;
	int newMixNum = MixNum + addN;
	int newL = SigmaL(newMixNum);
	double* newAlpha = new double[CodebookNum * newMixNum];
	double* newMu = new double[CodebookNum * newMixNum * FDim];
	double* newInvSigma = NULL;
	double* newDetSigma = NULL;
	if (!isSharedSigma()) {
		newInvSigma = new double[CodebookNum * newL];
		newDetSigma = new double[CodebookNum * newMixNum];
	}


	bool* shouldSplit = new bool[CodebookNum * MixNum];

	for (int i = 0; i < CodebookNum * MixNum; i++) {
		shouldSplit[i] = false;
	}

	int* sortIdx = new int[MixNum];
	for (int i = 0; i < CodebookNum; i++) {
		myBubbleSort(Alpha + i * MixNum, MixNum, sortIdx);
		for (int j = 0; j < addN; j++) {
			shouldSplit[i * MixNum + sortIdx[j]] = true;
		}
	}
	delete [] sortIdx;

	int T = isFullRank() ? FDim * FDim : FDim;
	int newIdx = 0;
	for (int i = 0; i < CodebookNum * oldMixNum; i++) {
		if (shouldSplit[i]) {
			newAlpha[newIdx] = Alpha[i] / 2;
			newAlpha[newIdx + 1] = Alpha[i] / 2;

			if (!isSharedSigma()) {
				newDetSigma[newIdx] = DetSigma[i];
				newDetSigma[newIdx + 1] = DetSigma[i];

				double* oldInvSigmaPtr = InvSigma + i * T;
				double* newInvSigmaPtr1 = newInvSigma + newIdx * T;
				double* newInvSigmaPtr2 = newInvSigmaPtr1 + T;
				memcpy(newInvSigmaPtr1, oldInvSigmaPtr, T * sizeof(double));
				memcpy(newInvSigmaPtr2, oldInvSigmaPtr, T * sizeof(double));
			}
			newIdx += 2;
		} else {
			newAlpha[newIdx] = Alpha[i];
			if (!isSharedSigma()) {
				newDetSigma[newIdx] = DetSigma[i];
				double* oldInvSigmaPtr = InvSigma + i * T;
				double* newInvSigmaPtr1 = newInvSigma + newIdx * T;
				memcpy(newInvSigmaPtr1, oldInvSigmaPtr, T * sizeof(double));

			}
			newIdx += 1;
		}
	}

	for (int i = 0; i < CodebookNum; i++) {
		double* newMuPtr0 = newMu + FDim * newMixNum * i;
		for (int j = 0; j < oldMixNum; j++) {
			double* oldMuPtr = Mu + FDim * oldMixNum * i + FDim * j;
			if (shouldSplit[i * MixNum + j]) {
				double* invSigmaPtr = NULL;
				if (isSharedSigma())
					invSigmaPtr = InvSigma + i * T;
				else
					invSigmaPtr = InvSigma + T * oldMixNum * i + T * j;

				double* newMuPtr1 = newMuPtr0;
				double* newMuPtr2 = newMuPtr1 + FDim;

				splitTwoMu(invSigmaPtr, oldMuPtr, newMuPtr1, newMuPtr2, offset);
				newMuPtr0 += FDim * 2;
			} else {
				memcpy(newMuPtr0, oldMuPtr, FDim * sizeof(double));
				newMuPtr0 += FDim;
			}
		}
	}

	delete [] Alpha;
	Alpha = newAlpha;
	delete [] Mu;
	Mu = newMu;

	if (!isSharedSigma()) {
		delete [] InvSigma;
		InvSigma = newInvSigma;

		delete [] DetSigma;
		DetSigma = newDetSigma;

	}

	MixNum = newMixNum;
}


bool GMMCodebookSet::mergeIsoCbs2BetaCbs(GMMCodebookSet* IsoCbset){
	if (CodebookNum != IsoCbset->CodebookNum || IsoCbset->cbType != CB_TYPE_FULL_RANK)
	{
		return false;
	}
	memcpy(Mu, IsoCbset->Mu, CodebookNum * MixNum * FDim * sizeof(double));
	memcpy(DurMean,IsoCbset->DurMean, sizeof(double) * CodebookNum);
	memcpy(DurVar, IsoCbset->DurVar,sizeof(double)* CodebookNum);
	memcpy(LogDurVar, IsoCbset->LogDurVar,sizeof(double)* CodebookNum);

	int IsoCbsetLen = IsoCbset->FDim * IsoCbset->FDim * MixNum;

	for (int i = 0; i < CodebookNum * MixNum; i++)
	{
		int idx = 0;
		for (int k = 0; k < FDim ;k++)
		{
			int j = 0;
			if (k <FEATURE_DIM)
			{

				for (j; j < FEATURE_DIM; j++)
				{
					InvSigma[i * SigmaL() * MixNum + idx++] = IsoCbset->InvSigma[j + i * IsoCbsetLen + k * FDim];
				}		
			}
			else
			{
				for (j = FEATURE_DIM; j< FDim; j++)
				{
					InvSigma[i * SigmaL() * MixNum + idx++] = IsoCbset->InvSigma[j + i * IsoCbsetLen + k * FDim];
				}
			}
		}
	}
	return true;

}

void GMMCodebookSet::writeCBset2File(const std::string& filename)
{

	FILE* fid = fopen(filename.c_str(),"wb");
	fwrite(&cbType,sizeof(int),1,fid);
	fwrite(&MixNum,sizeof(int),1,fid);
	float* alpha = new float[MixNum * CodebookNum];
	for (int i =0;i < MixNum * CodebookNum;i++)
	{
		alpha[i] = (float)Alpha[i]; 
	}
	float* mu = new float[MixNum * CodebookNum * FDim];
	for (int i =0;i < MixNum * CodebookNum*FDim;i++)
	{
		mu[i] = (float)Mu[i];
	}
	float* invSigmaCompress = (float*)malloc(CodebookNum * MixNum * FDim * (FDim + 1) / 2 * sizeof(float));
	for (int i = 0; i < CodebookNum*MixNum; i++) {
		double* invSigmaPtr = InvSigma + FDim * FDim * i;
		float* invSigmaCompressPtr = (invSigmaCompress + FDim * (FDim + 1) / 2 * i);
		for (int j = 0; j < FDim; j++) {
			for (int k = 0; k < j; k++) {
				invSigmaCompressPtr[j * (j + 1) / 2 + k] = (float)invSigmaPtr[j * FDim + k];
			}
			invSigmaCompressPtr[j * (j + 1) / 2 + j] = (float)invSigmaPtr[j * FDim + j] / 2;
		}
	}
	float* LogDetSigma=(float*)malloc(CodebookNum * MixNum * sizeof(float));
	for (int i = 0; i < CodebookNum * MixNum; i++)
	{
		LogDetSigma[i] = (float)log(DetSigma[i]);
	}
	fwrite(alpha,sizeof(float),MixNum *CodebookNum,fid);
	fwrite(mu,sizeof(float),MixNum*CodebookNum*FDim,fid);
	fwrite(invSigmaCompress,sizeof(float), CodebookNum * MixNum * FDim * (FDim + 1) / 2,fid);
	fwrite(LogDetSigma,sizeof(float),CodebookNum * MixNum,fid);
	fclose(fid);
	delete []alpha;
	delete []mu;
	free(invSigmaCompress);
	free(LogDetSigma);
}