#ifndef ZL_WORDLEVEL_MMIESTIMATOR_H
#define ZL_WORDLEVEL_MMIESTIMATOR_H

#include "GMMEstimator.h"

class CWordLevelMMIEstimator :public GMMEstimator
{
	double m_dDsm;
public:
	CWordLevelMMIEstimator(int fDim, int mixNum, int cbType, int maxIter, bool useCuda, std::vector<double>& WordGamma);
	~CWordLevelMMIEstimator(){}
	int estimate();
	void loadWordGamma(std::vector<double>& WordGamma);
private:
	bool singleGaussianMMIEupdate();
	std::vector<double>& mWordGamma;
};






#endif