#include "StateProbabilityMap.h"
#include <fstream>

CStateProbMap::CStateProbMap(int cbNum, double* outMap)
{
	CbNum = cbNum;
	if (!outMap)
	{
		StateProbMap = new double[CbNum * CbNum];
		memset(StateProbMap, 0, sizeof(double) * CbNum * CbNum);
	}
	else
	{
		StateProbMap = outMap;
	}
}

CStateProbMap::~CStateProbMap()
{
	if (StateProbMap)
	{
		delete [] StateProbMap;
		StateProbMap = NULL;
	}
}

bool CStateProbMap::pushToMap(GMMProbBatchCalc* gbc, std::vector<int>& label)
{

	for (int i = 0; i < label.size(); i++)
	{
		for (int s = 0; s < CbNum; s++)
		{
			double val = gbc->getStateLh(s, i);
			if (val != val)
			{
				std::cout<<"Fatal error in gbc ProbCalc"<<std::endl;
				return false;
			}
			 StateProbMap[label[i] * CbNum + s] += val;// Y data: X state;
		}
	}
	return true;
}


bool CStateProbMap::saveOutMaptoFile(std::string FileName, FrameWarehouse* fw)
{
	if (FileName == "")
	{
		FileName = "StateProbMap.txt";
	}
	std::ofstream fo;
	fo.open(FileName,std::ios::out);
	for (int i = 0; i < CbNum; i++)
	{
		double gen = MathUtility::logSumExp(StateProbMap + i * CbNum,CbNum);

		for (int j = 0; j < CbNum; j++)
		{
			double val = StateProbMap[i * CbNum + j] - gen;
			val = abs(val) > 308? 0 : exp(val);
			val = val > 0.0001? val :0;
			if (val != val)
			{
				std::cout<<"StateProbMap or Collected FrameNum value fatal!"<<std::endl;
				return false;
			}
			
			fo<<val<<" ";
		}
		fo<<std::endl;
	}
	fo.close();
	return true;
}

 bool CStateProbMap::mergeMapToMatrix(std::vector<std::vector<double>>& outMatrix,  FrameWarehouse* fw)
{
	double* p_dMax = new double[CbNum];
	for (int i = 0; i < CbNum; i++)
	{

		for (int j = 0; j < CbNum; j++)
		{
			StateProbMap[i * CbNum + j] /= fw->getFrameNum(i);
		}
// 		p_dMax[i] = -10000;
		double gen = 0;
		gen = MathUtility::logSumExp(StateProbMap + i * CbNum,CbNum);
		for (int j = 0; j < CbNum; j++)
		{
// 			double val = StateProbMap[i * CbNum + j] / fw->getFrameNum(i);
// 			if (val != val)
// 			{
// 				std::cout<<"StateProbMap or Collected FrameNum value fatal!"<<std::endl;
// 				return false;
// 			}
// 			p_dMax[i] = p_dMax[i] > val? p_dMax[i] : val;
			double val = StateProbMap[i * CbNum + j] - gen;
			outMatrix[i].push_back(val);
		}
	}

	/*for (int i = 0; i < CbNum; i++)
	{
// 		for (int j = 0; j < CbNum; j++)
// 		{
// 			outMatrix[i][j] -= p_dMax[i];
// 		}
		for (int j = 0; j < CbNum; j++)
		{
			double val = exp(outMatrix[i][j]);
			if (val > 0.000001)
			{
				outMatrix[i][j] = 1;
			}
			else
			{
				outMatrix[i][j] = 0;
			}
		}
	}*/
	delete []p_dMax;

	return true;

}

 void CStateProbMap::setFW(FrameWarehouse* f){
	 fw = f;
 }
 

