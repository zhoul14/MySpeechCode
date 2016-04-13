#include <iostream>
#include <string>
#include <stdio.h>
#include <vector>
#include "../GMMCodebook/GMMProbBatchCalc.h"
#include "../GMMCodebook/GMMCodebookSet.h"
#include "../CommonLib/FileFormat/FeatureFileSet.h"
#include "../CommonLib/Dict/WordDict.h"
//#include "boost/filesystem.hpp"
//#include "CTriPhoneDDBRec.h"
#include "../CommonLib/ReadConfig/RecParam.h"
#include "../NBestRecAlgorithm/NBestRecAlgorithm.h"
#include <direct.h>
#include <fstream>
#include <sstream>
#define Nbest 1
#define LongestAnserSize 10
using namespace std;
class WrongWordAnalysis
{

public:

	~WrongWordAnalysis();
	void ReadOneFile(string filename,string answerFileName);
	WrongWordAnalysis(string Configfilename,string outFileName,string statFile,string dictConfigFIle) ;
	int GetFileSize(string filename);
	void printWrongRecResToFile(FILE * fid);	
private:
	unsigned int SentenceNum;
	char* answerBuf;
	int WrongList[1254];
	void trimStr(string& inStr);
	void printWrongListToFile(FILE* fid);
	vector<vector<string>>wrongIdList;
	vector<vector<string>>InsertErrorIdList;
	vector<vector<string>>ConsonanErrorList;
	vector<vector<string>>VowelErrorList;
	vector<vector<int>>ErrorMatrix;
	WordDict* thisdict;
	vector<vector<vector<SWord>>> MyresList;	
	vector<int>PerFileErrorList;
	int FileIdx;
	void clearBuffer();
	int allCnt;
	int ProcessNum;
};
