#include <vector>
#include <iostream>
#include <stdio.h>
#include <string>
#include "../GMMCodebook/FrameWarehouse.h"
#include "../CommonLib//Math/MathUtility.h"
#include "../CommonLib/CudaCalc/CUGaussLh.h"
#include "../CommonLib/CudaCalc/CUDiagGaussLh.h"
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../CommonLib/CudaCalc/CUShareCovLh.h"
#include "../GMMCodebook/GMMCodebookSet.h"
class FrameWarehouse;
class GMMProbBatchCalc;
class FrameWarehouse;
class CStateProbMap
{
public:
	CStateProbMap(int cbnum, double* outMap = NULL);
	~CStateProbMap();
	bool pushToMap(GMMProbBatchCalc* gbc, std::vector<int>& label);
	bool mergeMapToMatrix(std::vector<std::vector<double>>& om, FrameWarehouse* fw);
	void setFW(FrameWarehouse* f);
	bool saveOutMaptoFile(std::string FileName, FrameWarehouse* fw);
	
private:
	int CbNum;
	FrameWarehouse* fw;
	double* StateProbMap;
};



