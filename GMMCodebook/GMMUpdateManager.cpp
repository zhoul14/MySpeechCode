#include "GMMUpdateManager.h"
#include "../CommonLib/Math/MathUtility.h"
#include <string>
#include <iostream>
#include <fstream>
#include "algorithm"
#include "sstream"
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../SpeechSegmentAlgorithm/SegmentAlgorithm.h"
#define  MAX_CNUM 7
//GMMEstimator(int fDim, int mixNum, int maxIter);

GMMUpdateManager::GMMUpdateManager(GMMCodebookSet* codebooks, int maxIter, WordDict* dict, double minDurSigma, const char* logFileName, bool cudaFlag, bool useSegmentModel, bool MMIEFlag = false, double Dsm = 0.0) {
	this->codebooks = codebooks;
	this->dict = dict;
	this->minDurVar = minDurSigma * minDurSigma;
	this->useSegmentModel = useSegmentModel;
	this->m_bMMIE = MMIEFlag;
	this->m_dDsm = Dsm;
	this->updateIter = 0;
	m_pMMIEres = new double*[codebooks->CodebookNum];
	m_pMMIEgen = new double*[codebooks->CodebookNum];

	m_WordGamma.resize(codebooks->CodebookNum);

	//reiniter = new KMeansReinit(10, 100);
	logFile = fopen(logFileName, "w");
	if (!logFile) {
		printf("cannot open log file[%s] in update manager\n", logFileName);
		exit(-1);
	}
	estimator = new CMMIEstimator(codebooks->getFDim(), codebooks->getMixNum(), codebooks->getCbType(), maxIter, cudaFlag, 0, codebooks->BetaNum);
	estimator->setOutputFile(logFile);

	std::string fwpath = "FWTMP";
	int maxFNum = 10000000;
	fw = new FrameWarehouse(fwpath, codebooks->getCodebookNum(), codebooks->getFDim(), maxFNum);

	int cbNum = codebooks->getCodebookNum();

	if (useSegmentModel) {
		firstMoment = new double[cbNum];
		memset(firstMoment, 0, cbNum * sizeof(double));

		firstMomentOfLog = new double[cbNum];
		memset(firstMomentOfLog, 0, cbNum * sizeof(double));

		secondMoment = new double[cbNum];
		memset(secondMoment, 0, cbNum * sizeof(double));

		durCnt = new int[cbNum];
		memset(durCnt, 0, cbNum * sizeof(int));
	}


	successUpdateTime = new int[cbNum];
	memset(successUpdateTime, 0, cbNum * sizeof(int));

}

int GMMUpdateManager::collect(const std::vector<int>& frameLabel, double* frames) {
	if (frameLabel.size() == 0) {
		return 0;
	}

	int fDim = codebooks->getFDim();
	int cbNum = codebooks->getCodebookNum();

	//统计duration的一阶矩和二阶矩
	if (useSegmentModel) {
		int currentCb = *(frameLabel.begin());

		if(currentCb>=cbNum)
			currentCb=cbNum-1;

		int currentDur = 0;
		for (auto i = frameLabel.begin(); i != frameLabel.end(); i++) {
			int cb = *i;

			if(cb>=cbNum)
				cb=cbNum-1;

			if (cb == currentCb) {
				currentDur++;
			} else {
				firstMoment[currentCb] += currentDur;
				firstMomentOfLog[currentCb] += log((double)currentDur);
				secondMoment[currentCb] += currentDur * currentDur;
				durCnt[currentCb]++;

				currentCb = cb;
				currentDur = 1;
			}
		}
		//处理最后一段duration
		firstMoment[currentCb] += currentDur;
		firstMomentOfLog[currentCb] += log((double)currentDur);
		secondMoment[currentCb] += currentDur * currentDur;
		durCnt[currentCb]++;
	}


	//统计各码本对应的帧
	int time = -1;
	for (auto i = frameLabel.begin(); i != frameLabel.end(); i++) {
		time++;

		double* ft = frames + fDim * time;
		int cbid = (*i);
		if(cbid>=cbNum)
			cbid=cbNum-1;
		fw->pushFrame(cbid, ft);
	}

	int totalFrameNum = fw->getTotalFrameNum();
	return totalFrameNum;


}


int GMMUpdateManager::collectWordGamma(const std::vector<int>& frameLabel, std::vector<SWord>& recLh, int ans, double segLh){
	if (frameLabel.size() == 0) {
		return 0;
	}

	bool ansInRec = false;
	int fDim = codebooks->getFDim();
	int cbNum = codebooks->getCodebookNum();
	for (auto r : recLh)if (r.wordId == ans)ansInRec = true;
	if (!ansInRec)
	{
		SWord s;
		s.wordId = ans;
		s.lh = segLh;
		s.jumpTime[0]=0;

		for (int i = 0; i != 6; i++)
		{
			int t = frameLabel[s.jumpTime[i]];
			for (int j = s.jumpTime[i]; j != frameLabel.size(); j++)
			{
				if(j)
				if (frameLabel[j]!=t)
				{
					s.jumpTime[i+1] = j;
					break;
				}				
			}
		}
		recLh.push_back(s);
	}
	int len = recLh.size();


	int time = -1;
	double* pLh = new double[len];
	int* pWordId = new int[len];
	for (int i = 0; i != len; i++)
	{
		pWordId[i] = recLh[i].wordId;
		pLh[i] = recLh[i].lh / 10;
		recLh[i].jumpTime[6] = frameLabel.size();
	}

	double sumLh = MathUtility::logSumExp(pLh, len);

	for (auto i = frameLabel.begin(); i != frameLabel.end(); i++) {
		time++;

		int cbid = (*i);
		if(cbid>=cbNum)
			cbid=cbNum-1;
		if (cbid == dict->getNoiseCbId())
		{
			continue;
		}
		// 		if (i != frameLabel.begin()&& cbid == *(i - 1))
		// 		{
		// 			m_WordGamma[cbid].push_back(m_WordGamma[cbid].back());
		// 		}
		// 		else
		
		auto vec = dict->getstateUsingWord(cbid);
		vector<double> u;
		for (int j = 0; j != vec.size(); j++)
		{
			auto p = find(pWordId, pWordId + len, vec[j]);
			if(p != pWordId + len)
			{
				int it = dict->getCbType(cbid);
				int beginTime = recLh[p - pWordId].jumpTime[it];
				int endTime = recLh[p - pWordId].jumpTime[it + 1];
				if (time >= beginTime && time< endTime)
				{
					u.push_back(pLh[p - pWordId]);
				}
			}
			//t[j] = recLh[ans] - sumLh;
		}
		double* t = new double[u.size()];
		for (int k = 0; k != u.size(); k++)
		{
			t[k] = u[k];
		}
		m_WordGamma[cbid].push_back(MathUtility::logSumExp(t, u.size()) - sumLh);
		delete []t;

	}
	delete []pWordId;
	delete []pLh;
	return ansInRec;
}
int GMMUpdateManager::collectWordGamma(const std::vector<int>& frameLabel, std::vector<SegmentResult>& recLh)
{
	if (frameLabel.size() == 0) {
		return 0;
	}

	bool ansInRec = false;
	int fDim = codebooks->getFDim();
	int cbNum = codebooks->getCodebookNum();
	int len = recLh.size();


	int time = -1;
	double* pLh = new double[len];
	int* pWordId = new int[len];
	for (int i = 0; i != len; i++)
	{
		pWordId[i] = i;
		pLh[i] = recLh[i].lh / 20;
	}
	double sumLh = MathUtility::logSumExp(pLh, len);
	for (auto i = frameLabel.begin(); i != frameLabel.end(); i++) {
		time++;

		int cbid = (*i);
		if(cbid>=cbNum)
			cbid=cbNum-1;
		if (cbid == dict->getNoiseCbId())
		{
			continue;
		}
		auto cbidUsingWords = dict->getstateUsingWord(cbid);
		vector<double> u;
		for (int j = 0; j != cbidUsingWords.size(); j++)
		{
			int CBUsingWordIdx = cbidUsingWords[j];
			if(recLh.at(CBUsingWordIdx).frameLabel[time] == cbid)
			{
				u.push_back(pLh[CBUsingWordIdx]);
			}
		}
		double* t = new double[u.size()];
		for (int k = 0; k != u.size(); k++)
		{
			t[k] = u[k];
		}
		m_WordGamma[cbid].push_back(MathUtility::logSumExp(t, u.size()) - sumLh);
		delete []t;
	}
	delete []pLh;
	delete []pWordId;
	return true;
}


GMMUpdateManager::~GMMUpdateManager() {

	fclose(logFile);
	delete estimator;
	fw->clearFrames();
	delete fw;
	if (useSegmentModel) {
		delete [] firstMoment;
		delete [] firstMomentOfLog;
		delete [] secondMoment;
		delete [] durCnt;
	}

	delete []m_pMMIEgen;
	delete []m_pMMIEres;
	delete [] successUpdateTime;
}

//根据EM算法的结果更新码本，并清空之前累积的帧
std::vector<int> GMMUpdateManager::update() {
	using std::vector;

	int cbNum = codebooks->getCodebookNum();
	int fDim = codebooks->getFDim();
	double sumMMIEval = 0.0;

	int* cbFrameCnt = new int[cbNum];

	int totalFrameCnt = 0;
	for (int i = 0; i < cbNum; i++) {
		cbFrameCnt[i] = fw->getFrameNum(i);
		totalFrameCnt += cbFrameCnt[i];
	}
	printf("totalFrameCnt:[%d]\n",totalFrameCnt);

	vector<int> res;
	estimator->getDataMat(m_DataMatrix);
	GMMCodebook** cbs = new GMMCodebook*[cbNum];

	for (int i = 0; i < cbNum; i++) 
	{
		printf("updating codebook %d, %d samples\n", i, cbFrameCnt[i]);
		fprintf(logFile, "updating codebook %d, %d samples\n", i, cbFrameCnt[i]);

		if (cbFrameCnt[i] == 0) {
			int errCode = GMMEstimator::SAMPLE_NOT_ENOUGH;
			std::string t = GMMEstimator::errorInfo(errCode);
			printf("--- UPDATE FAIL: %s ---\n", t.c_str());
			fprintf(logFile, "--- UPDATE FAIL: %s ---\n", t.c_str());
			res.push_back(errCode);
			continue;
		}
		GMMCodebook cb (codebooks->getCodebook(i));
		cbs[i] = new GMMCodebook(cb);
		GMMCodebook* cbp = cbs[i];

		estimator->setCbId(i);

		int errCode = GMMEstimator::SUCCESS;
		if (useSegmentModel) {
			//高斯分布建模duration
			cb.DurMean = firstMoment[i] / durCnt[i];
			cb.DurVar = secondMoment[i] / durCnt[i] - cb.DurMean * cb.DurMean;

			if (cb.DurVar < minDurVar) {
				cb.DurVar = minDurVar;
			}
		}

		if (m_bMMIE && !m_ConMatrix.empty() && m_ConMatrix[i].size() > 1)
		{
			int len = m_ConMatrix[i].size();
			int selfIdx;
			double** allFramesOfCbs = new double*[len];
			estimator->initMMIE(len, m_dDsm);
			estimator->loadParam(cb.Alpha, cb.Mu, cb.InvSigma, NULL);
			estimator->setSelfIdx(-1);


			//estimator->initMMIECbs(m_DataMatrix[i].size());
			//for (int l = 0; l != m_DataMatrix[i].size(); l++)
			//{
			//	GMMCodebook cbl (codebooks->getCodebook(m_DataMatrix[i][l] - 1));
			//	estimator->loadMMIECbs(cbl.Alpha, cbl.Mu, cbl.InvSigma, l);
			//}//
			for (int ite = 0; ite < /*estimator->getMaxIter()*/1; ite++)
			{
				for (int k = 0; k != len; k++)
				{
					int dataId = m_ConMatrix[i][k]-1;

					if (ite == 0)
					{
						allFramesOfCbs[k] = new double[fDim * cbFrameCnt[dataId]];
						fw->loadFrames(dataId , allFramesOfCbs[k]);
						if(dataId == i) estimator->setSelfIdx(k);
						estimator->loadMMIEData(allFramesOfCbs[k], cbFrameCnt[dataId], k, dataId);
					}

					if (find(m_DataMatrix[dataId].begin(), m_DataMatrix[dataId].end(), i + 1) != m_DataMatrix[dataId].end())
					{

						estimator->initMMIECbs(m_DataMatrix[dataId].size());
						for (int l = 0; l != m_DataMatrix[dataId].size(); l++)
						{
							GMMCodebook cbl (codebooks->getCodebook(m_DataMatrix[dataId][l] - 1));
							estimator->loadMMIECbs(cbl.Alpha, cbl.Mu, cbl.InvSigma, l);
						}

						estimator->prepareGammaIForUpdate(k, ite == 0);

						estimator->destroyMMIECbs();
					}
					else
					{
						estimator->setGamma1(k);
					}

				}

				double res;
				bool bConv = estimator->checkGamma(res);
				printf("res:[%lf]\n",res);

				if (/*res - resHistory < 0 ||*/ abs(res) < abs(-10e-6) /*||resHistory > -10e-4*/)
				{
					errCode = GMMEstimator::ILL_CONDITIONED;
					estimator->printfGamma(logFile);
					printf("res:[%lf],too small\n",res);
					break;
				}
				else 
				{

					// 					codebooks->updateCodebook(i, cb);
					*cbp = cb;
					//if (res < -0.98)
					{
						estimator->printfGamma(logFile);
					}
				}
				if (!bConv)
				{
					printf("res:[%lf]\n",res);
					sumMMIEval += res;
					errCode = estimator->estimate();
					if (errCode != GMMEstimator::SUCCESS)
					{
						break;
					}
					estimator->saveParam(cb.Alpha, cb.Mu, cb.InvSigma);
					//codebooks->updateCodebook(i, cb);

				}
				else
				{
					errCode = GMMEstimator::ILL_CONDITIONED;
					break;
				}
			}			

			//estimator->destroyMMIECbs();

			res.push_back(errCode);
			estimator->destructMat(allFramesOfCbs, len);	
			delete []allFramesOfCbs;
			estimator->destoyMMIEr();
		}
		else if(!updateIter || !m_bMMIE)
		{

			double* allFramesOfCb = new double[cbFrameCnt[i] * fDim];

			fw->loadFrames(i, allFramesOfCb);
			bool mixFeature = false;
			if (cb.cbType == GMMCodebook::CB_TYPE_FULL_MIX)
			{
				mixFeature = true;
			}
			estimator->loadParam(cb.Alpha, cb.Mu, cb.InvSigma, mixFeature ? cb.Beta : NULL);
			estimator->loadData(allFramesOfCb, cbFrameCnt[i]);
			errCode = ((GMMEstimator*)estimator)->estimate();
			res.push_back(errCode);

			delete [] allFramesOfCb;
		}

		if(m_bMMIE)
		{
			fprintf(logFile," m_ConMatrix[i][j]:");

			for (int j = 0; j != m_ConMatrix[i].size(); j++)
			{
				fprintf(logFile,"[%d] ", m_ConMatrix[i][j]);

			}
			fprintf(logFile,"\n");
		}

		if (errCode != GMMEstimator::SUCCESS) {
			successUpdateTime[i] = 0;
			std::string t = GMMEstimator::errorInfo(errCode);
			printf("--- UPDATE FAIL: %s ---\n", t.c_str());
			fprintf(logFile, "--- UPDATE FAIL: %s ---\n", t.c_str());
			continue;
		}

		if(cb.cbType == GMMCodebook::CB_TYPE_FULL_MIX)
			estimator->saveParam(cb.Alpha, cb.Mu, cb.InvSigma,cb.Beta);
		else 
			estimator->saveParam(cb.Alpha, cb.Mu, cb.InvSigma);
		*cbp = cb;
		successUpdateTime[i]++;
		//estimator->getDataMat(m_DataMatrix);

	}
	for (int i = 0; i < cbNum; i++)
	{
		codebooks->updateCodebook(i, *cbs[i]);
		delete cbs[i];
	}
	delete [] cbs;

	if(m_bMMIE)
	{
		for (int i = 0; i < cbNum ; i++)
		{
			updateSMLmat(i);
		}
		makeConvMatrix();

		fprintf(logFile,"______________sum MMIE Objective Value:[%lf]___________\n",sumMMIEval);
	}
	//fw->clearFrames();
	delete [] cbFrameCnt;
	// 	for (int i = 0; i != codebooks->CodebookNum; i++)
	// 	{
	// 		delete []m_pMMIEres[i];
	// 		delete []m_pMMIEgen[i];
	// 	}
	updateIter++;

	return res;
}

int GMMUpdateManager::getSuccessUpdateTime(int cbidx) const {
	return successUpdateTime[cbidx];
}

bool GMMUpdateManager::getMMIEmatrix(std::string filename, bool bDataState)
{
	fstream ifs(filename,ios::in);

	std::string line;

	int cnt = 0;

	std::vector<std::vector<int>>& M = bDataState ? m_DataMatrix : m_ConMatrix;

	while (getline(ifs,line))
	{
		std::stringstream ss(line);
		vector<int>vec;
		int val;
		while (ss>>val)
		{			
			vec.push_back(val);
		}
		M.push_back(vec);
	}
	return true;
}

int GMMUpdateManager::setMMIEmatrix(std::vector<std::vector<double>>&x)
{
	setSimilarMatrix(x);
	return makeConvMatrix();
}

int GMMUpdateManager::makeConvMatrix(){
	int cnt = 0;
	int cnt2 = 0;
	int num = SimilarMat.size();
	if (!m_ConMatrix.empty())m_ConMatrix.clear();
	if (!m_DataMatrix.empty())m_DataMatrix.clear();

	double* dataCon = new double[num * num];
	double* stateCon = new double[num * num];

	double* tempVec1 = new double[num];
	double* tempVec2 = new double[num];

	for (int i = 0; i != num; i++)
	{
		for (int j = 0; j != num; j++)
		{
			dataCon[i * num + j] = SimilarMat[i][j];
			stateCon[i * num + j] = SimilarMat[j][i];
		}
		double dGen = MathUtility::logSumExp( dataCon + i * num, num);
		double sGen = MathUtility::logSumExp( stateCon + i * num, num);
		for (int j = 0; j != num; j++)
		{
			dataCon[i * num + j] -= dGen;
			stateCon[i * num + j] -= sGen;
		}

	}


	for (int i = 0; i != num; i++)
	{
		vector<int>vec;
		vector<int>vec2;
		if (i != codebooks->CodebookNum - 1)
		{
			memcpy(tempVec1, dataCon + i * num, sizeof(double) * num);
			memcpy(tempVec2, stateCon + i * num, sizeof(double) * num);
			sort(tempVec1, tempVec1 + num);
			sort(tempVec2, tempVec2 + num);

			for (int j = 0; j < num; j++)
			{
				if (dataCon[i * num + j] >= tempVec1[num - MAX_CNUM] || j == i)
				{
					fprintf(logFile,"D%d ", j + 1);
					vec.push_back(j + 1);
					cnt += 1;
				}
				if(stateCon[i * num + j] >= tempVec2[num - MAX_CNUM] || j == i)
				{
					fprintf(logFile,"C%d ", j + 1);
					vec2.push_back(j + 1);
					cnt2 += 1;				
				}

			}
			fprintf(logFile,"\n");
		}
		else{
			vec.push_back(i);
			vec2.push_back(i);

		}
		m_DataMatrix.push_back(vec);
		m_ConMatrix.push_back(vec2);

	}
	printf("prime date conv cnt[%d] state cnt [%d]!!!!\n",cnt,cnt2);
	fprintf(logFile,"conv cnt[%d] state cnt [%d]!!!!\n",cnt,cnt2);


	delete []dataCon;
	delete []stateCon;
	delete []tempVec1;
	delete []tempVec2;	
	return cnt;
}

// int GMMUpdateManager::makeConvMatrix(){
// 	{
// 		vector<vector<int>> v = m_ConMatrix;
// 
// 		int cnt =0;
// 		bool* flagList = new bool[v.size()];
// 		memset(flagList, 0 , v.size());
// 		{
// 			for (int i = 0; i != v.size(); i++)
// 			{
// 				if (flagList[i])
// 				{
// 					continue;
// 				}
// 				vector<int> vec = v[i];
// 				int it = 0;
// 				cnt = 1;
// 				while(cnt)				
// 				{
// 					cnt = 0;
// 					while (it < vec.size())
// 					{
// 
// 						for (int j = 0; j != v[vec[it]].size(); j++)
// 						{
// 							if (find(vec.begin(), vec.end(), v[vec[it]][j]) == vec.end())
// 							{
// 								cnt ++;
// 								vec.push_back(v[vec[it]][j]);
// 							}
// 						}
// 						it++;
// 					}
// 
// 				}
// 
// 				mergeIntoVec(vec, v, flagList);
// 			}
// 
// 		}
// 		cnt =0;
// 		for (auto i = v.begin(); i != v.end(); i++)
// 		{
// 			sort((*i).begin(),(*i).end());
// 			for (auto j = (*i).begin(); j != (*i).end(); j++)
// 			{
// 				(*j)++;
// 			}
// 			cnt += (*i).size();
// 		}
// 		m_ConMatrix = v;
// 		m_DataMatrix = v;
// 		printf("cnt:update[%d]",cnt);
// 		return cnt;
// 	}
// }


void GMMUpdateManager::mergeIntoVec(vector<int> vec, vector<vector<int>>&v, bool* bList)
{
	for (auto i = vec.begin(); i != vec.end(); i++)
	{
		v[*i] = vec;
		bList[*i] = true;
	}
}

void GMMUpdateManager::updateSMLmat(int idx){
	int n = fw->getFrameNum(idx);
	double * buff = new double[codebooks->getFDim() * n];
	fw->loadFrames(idx,buff);
	vector<double>x;
	gbc->calcSimularity(buff,n,x);
	SimilarMat[idx] = x;
	//makeConvMatrix();
	delete []buff;
}

void GMMUpdateManager::prepareMMIELh()
{
	int cbNum = codebooks->CodebookNum;
	int fDim = codebooks->FDim;
	for (int i = 0; i != codebooks->CodebookNum; i++)
	{
		int fnum = fw->getFrameNum(i);
		m_pMMIEres[i] = new double[fnum * cbNum];
		m_pMMIEgen[i] = new double [fnum];
		double* features = new double[fnum* fDim];
		gbc->CalcAlphaWeightLh(features, fnum, m_pMMIEres[i], m_pMMIEgen[i]);
		delete []features;
	}

}