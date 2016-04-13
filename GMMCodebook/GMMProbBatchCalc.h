#ifndef WJQ_GMM_PROB_BATCH_CALC_H
#define WJQ_GMM_PROB_BATCH_CALC_H
#include "../CommonLib/CudaCalc/CUGaussLh.h"
#include "GMMCodebookSet.h"
#include <iostream>
#include <cuda.h>
#include "vector"

#define LOG_2_PI 1.837877066409345

class GMMProbBatchCalc {

private:
	const GMMCodebookSet* codebooks;

	bool useCuda;

	bool useSegmentModel;

	double* alphaWeightedStateLh;

	bool* mask;

	int lhBufLen;

	void* unmaskedCalc;

	double m_dCoef;

	//预计算每个码本dur=1:200的duration似然值
	const static int PRECALCULATED_DURATION_LENGTH = 400;

	static double normlh(double x, double mean, double var, double logvar)	{
		if (var < 1e-10) {
			printf("error, var[%f] should be positive\n", var);
			exit(-1);
		}
		double res = -0.5 * (LOG_2_PI + logvar + 1 / var * (x - mean) * (x - mean));
		return res;
	}

	//计算LogSumExp函数
	double logOfSumExp(double* componentLh, double* alpha, int n);

	double gausslh(double* x, double* mean, double* invsigma, double cst);

	double durWeight;

	void preparePreDurLh();

public:
	const static int CUDA_AUTO = -1;
	const static int CUDA_OFF = 0;
	const static int CUDA_ON = 1;

	double* preDurLh;

	GMMProbBatchCalc(const GMMCodebookSet* cb, bool useCuda, bool useSegmentModel, double coef = 0);

	void setCodebookSet(const GMMCodebookSet* cb);

	void calcSimularity(double *features, int fnum, std::vector<double>& vec);

	void CalcAlphaWeightLh(double *features, int fnum,double * res, double* resGen = NULL);

	//使用cuda做预计算
	void prepare(double* features, int fnum);

	inline double getStateLh(int cbidx, int frameidx) {
		double p = alphaWeightedStateLh[frameidx * codebooks->CodebookNum + cbidx];
		return p;
	}
	inline double getDurLh(int cbidx, int dur) {
		if (!useSegmentModel)
			return 0;

		if (dur <= PRECALCULATED_DURATION_LENGTH) {
			return preDurLh[cbidx * PRECALCULATED_DURATION_LENGTH + dur - 1];
		} else {
			return normlh(dur, codebooks->DurMean[cbidx], codebooks->DurVar[cbidx], codebooks->LogDurVar[cbidx]) * durWeight;
		}
	}

	inline double getDurLhDelta(int cbidx, int dur) {
		if (!useSegmentModel)
			return 0;

		if (dur <= PRECALCULATED_DURATION_LENGTH) {
			double lh1 = preDurLh[cbidx * PRECALCULATED_DURATION_LENGTH + dur - 1];
			double lh2 = preDurLh[cbidx * PRECALCULATED_DURATION_LENGTH + dur - 2];
			return lh1 - lh2;
		} else {
			double lh1 = normlh(dur, codebooks->DurMean[cbidx], codebooks->DurVar[cbidx], codebooks->LogDurVar[cbidx]) * durWeight;
			double lh2 = normlh(dur - 1, codebooks->DurMean[cbidx], codebooks->DurVar[cbidx], codebooks->LogDurVar[cbidx]) * durWeight;
			return lh1 - lh2;
		}
	}

	void setDurWeight(double w);

	double getDurWeight();

	void setMask(bool* mask);

	~GMMProbBatchCalc();

	bool ifUseSegmentModel() {return useSegmentModel;}
};

#endif