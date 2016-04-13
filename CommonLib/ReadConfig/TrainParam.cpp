#include "TrainParam.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <windows.h>


using std::string;

void TrainParam::checkExist(const string& str) {
	checkExist(str.c_str());
}

int TrainParam::getFdim(){
	return fDim;
}

void TrainParam::checkExist(const char* str) {
	if (GetFileAttributes(str) == INVALID_FILE_ATTRIBUTES) {
		printf("path %s doesn't exists\n", str);
		exit(-1);
	}
	return;
}

void TrainParam::trim(string& s) {
	const char* endstr = " \r\n\t";
	s.erase(s.find_last_not_of(endstr)+1);
	s.erase(0, s.find_first_not_of(endstr));
}


TrainParam::TrainParam(const char* filename) {
// 	printf("pwd = %s\n", current_path().string().c_str());
 	checkExist(filename);
	std::ifstream infile(filename);
	if (!infile) {
		printf("can't open train config file [%s]\n", filename);
		exit(-1);
	}

	bool startParam = false;
	std::string line;
	while (getline(infile, line)) {
		trim(line);
		if (line == "")
			continue;

		if (line == "PARAM:") {
			startParam = true;
			break;
		}
	}
	if (!startParam) {
		printf("PARAM label not found in config file\n");
		exit(-1);
	}

	bool startData = false;
	std::unordered_map<std::string, std::string> ph;
	
	while (getline(infile, line)) {
		trim(line);
		if (line == "") {
			continue;
		}
		if (line == "DATA:") {
			startData = true;
			break;
		}
		std::stringstream ss(line);
		string pname, pvalue;
		ss >> pname >> pvalue;
		trim(pname);
		trim(pvalue);
		std::transform(pname.begin(), pname.end(), pname.begin(), ::toupper);
		ph[pname] = pvalue;
	}
	if (!startData) {
		printf("DATA label not found in config file\n");
		exit(-1);
	}
	
	while (getline(infile, line)) {
		trim(line);
		if (line == "")
			continue;

		std::stringstream ss(line);
		TSpeechFile f;
		ss >> f.featureFile >> f.answerFile >> f.maskFile;
		trim(f.answerFile);
		trim(f.featureFile);
		trim(f.maskFile);
		fileSets.push_back(f);
	}
	infile.close();

	outputCodebook = ph["OUTCODEBOOK"];
	if (outputCodebook == "") {
		printf("must assign output codebook path\n");
		exit(-1);
	}
	

	initCodebook = ph["INITCODEBOOK"];
	checkExist(initCodebook);

	wordDictFileName = ph["DICTCONFIG"];
	checkExist(wordDictFileName);

	useSegmentModel = ph["SEGMENTMODEL"] == "" ? 1 : stoi(ph["SEGMENTMODEL"]);
	useCuda = ph["USECUDA"] == "" ? 1 : stoi(ph["USECUDA"]);
	durWeight = ph["DURWEIGHT"] == "" ? 0 : stod(ph["DURWEIGHT"]);

	fDim = ph["FDIM"] == ""? 45 : stoi(ph["FDIM"]);
	if(fDim <= 0) {
		printf("fDim = %d, must > 0\n",fDim);
		exit(-1);
	}

	Coef = ph["COEF"] == ""? 0 : stoi(ph["COEF"]);
	if (Coef < 0){
		printf("coef = %f, must >= 0",Coef);
	}

	triPhone = ph["TRIPHONE"] =="" ? 1:stoi(ph["TRIPHONE"]); 
	if(triPhone!=true&&triPhone!=false){
		printf("TRIPHONE = %d, must be 1 or 0",triPhone);
		exit(-1);
	}

	trainNum = ph["TRAINNUM"] == "" ? 0 : stoi(ph["TRAINNUM"]);
	if (trainNum <= 0) {
		printf("trainNum = %d, must > 0\n", trainNum);
		exit(-1);
	}
	if (fileSets.size() > trainNum) {
		fileSets.resize(trainNum);
	}

	trainIterNum = ph["TRAINITER"] == "" ? 0 : stoi(ph["TRAINITER"]);
	if (trainIterNum <= 0) {
		printf("trainIterNum = %d, must > 0\n", trainIterNum);
		exit(-1);
	}

	EMIterNum = ph["EMITER"] == "" ? 0 : stoi(ph["EMITER"]);
	if (EMIterNum <= 0) {
		printf("EMIterNum = %d, must > 0\n", EMIterNum);
		exit(-1);
	}

	splitAddN = ph["SPLITADDN"] == "" ? 0 : stoi(ph["SPLITADDN"]);
	splitOffset = ph["SPLITOFFSET"] == "" ? 0 : stod(ph["SPLITOFFSET"]);
	if (splitAddN > 0 && splitOffset == 0) {
		printf("splitAddN[%d] > 0 but splitOffset = 0\n", splitAddN);
		exit(-1);
	}

	minDurSigma = ph["MINDURSIGMA"] == "" ? 1 : stod(ph["MINDURSIGMA"]);

	logPath = string("log");
	if (GetFileAttributes("log") == INVALID_FILE_ATTRIBUTES) {
		CreateDirectory("log", NULL);
	}
// 	using boost::property_tree::ptree;
// 	using boost::algorithm::trim;


// 	using std::string;
// 	ptree pt;
// 	read_xml(filename, pt);
// 
// 	outputCodebook = pt.get<string>("trainconfig.outcodebook");
// 	trim(outputCodebook);
// 
// 	initCodebook = pt.get<string>("trainconfig.initcodebook");
// 	trim(initCodebook);
// 	checkExist(initCodebook);
// 
// 	
// 	wordDictFileName = pt.get<string>("trainconfig.dictconfig");
// 	trim(wordDictFileName);
// 	checkExist(wordDictFileName);
// 
// 	
// 
// 	useSegmentModel = pt.get<int>("trainconfig.segmentmodel", 1);
// 
// 	useCuda = pt.get<int>("trainconfig.usecuda", 1);
// 
// 	durWeight = pt.get<int>("trainconfig.durweight");
// 
// 	trainNum = pt.get<int>("trainconfig.trainnum");
// 
// 	trainIterNum = pt.get<int>("trainconfig.trainiter");
// 
// 	EMIterNum = pt.get<int>("trainconfig.emiter");
// 
// 	splitAddN = pt.get<int>("trainconfig.splitaddn", 0);
// 	splitOffset = pt.get<double>("trainconfig.splitoffset", 0);
// 	if (splitAddN > 0 && splitOffset == 0) {
// 		printf("splitAddN[%d] > 0 but splitOffset = 0\n", splitAddN);
// 		exit(-1);
// 	}
// 
// 	
// 
// 	minDurSigma = pt.get<double>("trainconfig.mindursigma", 1);
// 
// 	logPath = string("log");
// 	if (!exists(logPath)) {
// 		create_directory(logPath);
// 	}
// 
// 	int cnt = 0;
// 	foreach_(ptree::value_type& v, pt.get_child("trainconfig.data")) {
// 		TSpeechFile f;
// 		ptree t = v.second;	//speech tag
// 		
// 		string s;
// 		s = t.get<string>("feature");
// 		trim(s);
// 		checkExist(s);
// 		f.featureFile = s;
// 
// 		s = t.get<string>("mask");
// 		trim(s);
// 		checkExist(s);
// 		f.maskFile = s;
// 
// 		s = t.get<string>("answer");
// 		trim(s);
// 		checkExist(s);
// 		f.answerFile = s;
// 
// 		fileSets.push_back(f);
// 		cnt++;
// 		if (cnt == trainNum)
// 			break;
// 	}
 }

int TrainParam::getSplitAddN() {
	return splitAddN;
}
std::string TrainParam::getInitCodebook() {
	return initCodebook;
}

bool TrainParam::getUseCudaFlag() {
	return useCuda != 0;
}

std::string TrainParam::getOutputCodebook() {
	return outputCodebook;
}

double TrainParam::getDurWeight() {
	return durWeight;
}

double TrainParam::getMinDurSigma() {
	return minDurSigma;
}

bool TrainParam::getTriPhone(){
	return triPhone;
}

double TrainParam::getSplitOffset() {
	return splitOffset;
}

std::string TrainParam::getWordDictFileName() {
	return wordDictFileName;
}

bool TrainParam::getSegmentModelFlag() {
	return (useSegmentModel != 0);
}

int TrainParam::getTrainNum() {
	return trainNum;
}

std::vector<TSpeechFile> TrainParam::getTrainFiles() {
	return fileSets;
}

std::string TrainParam::getLogPath() {
	return logPath;
}

int TrainParam::getEMIterNum() {
	return EMIterNum;
}

int TrainParam::getTrainIterNum() {
	return trainIterNum;
}

double TrainParam::getDCoef() {
	return Coef;
}