#include "RecParam.h"
#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
//#include <boost/property_tree/ptree.hpp>
//#include <boost/property_tree/xml_parser.hpp>
//#include <boost/foreach.hpp>
//#include <boost/algorithm/string.hpp>
//#include <boost/filesystem.hpp>

//#define foreach_ BOOST_FOREACH
void RecParam::checkExist(const char* str) {
	if (GetFileAttributes(str) == INVALID_FILE_ATTRIBUTES) {
		printf("path %s doesn't exists\n", str);
		exit(-1);
	}
	return;
}

void RecParam::checkExist(const std::string& str) {
	checkExist(str.c_str());
}

void RecParam::trim(std::string& s) {
	const char* endstr = " \r\n\t";
	s.erase(s.find_last_not_of(endstr)+1);
	s.erase(0, s.find_first_not_of(endstr));
}

RecParam::RecParam(char* filename) {
	checkExist(filename);
	std::ifstream infile(filename);
	if (!infile) {
		printf("can't open rec config file [%s]\n", filename);
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
	std::unordered_map<std::string, std::string> config;
	
	while (getline(infile, line)) {
		trim(line);
		if (line == "") {
			continue;
		}
		if (line == "DATA:") {
			startData = true;
			break;
		}
		std::string pname, pvalue;
		std::stringstream ss(line);
		ss >> pname >> pvalue;
		printf("pname = %s, pvalue = %s\n", pname.c_str(), pvalue.c_str());
		trim(pname);
		trim(pvalue);
		std::transform(pname.begin(), pname.end(), pname.begin(), ::toupper);
		config[pname] = pvalue;
	}
	if (!startData) {
		printf("DATA label not found in config file\n");
		exit(-1);
	}
	std::string ansFile, ftFile, maskFile, resFile;
	while (getline(infile, line)) {
		trim(line);
		if (line == "")
			continue;

		std::stringstream ss(line);
		RSpeechFile f;
		ss >> f.featureFile >> f.answerFile >> f.maskFile >> f.recResultFile;
		trim(f.answerFile);
		trim(f.featureFile);
		trim(f.maskFile);
		trim(f.recResultFile);
		fileSets.push_back(f);
	}
	infile.close();

	codebookFileName = config["CODEBOOK"];
	//checkExist(codebookFileName);(crossfit)

	wordDictFileName = config["DICTCONFIG"];
	checkExist(wordDictFileName);

	recNum = std::stoi(config["RECNUM"]);
	if (recNum <= 0) {
		printf("RECNUM = %d, must > 0\n", recNum);
		exit(-1);
	}
	if (recNum < fileSets.size())
		fileSets.resize(recNum);

	durWeight = std::stod(config["DURWEIGHT"]);
	multiJump = config["MULTIJUMP"] == "" ? -1 : std::stoi(config["MULTIJUMP"]);
	useCuda = config["USECUDA"] == "" ? 1 : std::stoi(config["USECUDA"]);
	useSegmentModel = config["SEGMENTMODEL"] == "" ? 1 : std::stoi(config["SEGMENTMODEL"]);
	fDim = config["FDIM"] == "" ? 45 :std::stoi(config["FDIM"]);
	triPhone = config["TRIPHONE"] == "" ? 1 :std::stoi(config["TRIPHONE"]);
	m_bHeadNOise = config["HEADNOISE"] == "" ? false : std::stoi(config["HEADNOISE"]);

	Coef = config["COEF"] == ""? 0 :std::stod(config["COEF"]);
	if (Coef < 0)
	{
		printf("coef = %f, must >= 0",Coef);
	}

	//using boost::property_tree::ptree;
	//using boost::algorithm::trim;
	//using std::string;
// 	ptree pt;
// 	read_xml(filename, pt);
// 
// 	codebookFileName = pt.get<string>("recconfig.codebook");
// 	trim(codebookFileName);
// 
// 	wordDictFileName = pt.get<string>("recconfig.dictconfig");
// 	trim(wordDictFileName);
// 
// 	recNum = pt.get<int>("recconfig.recnum");
// 	durWeight = pt.get<int>("recconfig.durweight");
// 	multiJump = pt.get<int>("recconfig.multijump", -1);
// 	useCuda = pt.get<int>("recconfig.usecuda", 1);
// 	useSegmentModel = pt.get<int>("recconfig.segmentmodel", 1);
// 
// 	int cnt = 0;
// 	foreach_(ptree::value_type& v, pt.get_child("recconfig.data")) {
// 		RSpeechFile f;
// 		ptree t = v.second;	//speech tag, v.first is "<set>"
// 
// 		string s;
// 		s = t.get<string>("feature");
// 		trim(s);
// 		f.featureFile = s;
// 
// 		s = t.get<string>("mask");
// 		trim(s);
// 		f.maskFile = s;
// 
// 		s = t.get<string>("answer");
// 		trim(s);
// 		f.answerFile = s;
// 
// 		s = t.get<string>("result");
// 		trim(s);
// 		f.recResultFile = s;
// 
// 		fileSets.push_back(f);
// 		cnt++;
// 		if (cnt == recNum)
// 			break;
// 	}
}

bool RecParam::getSegmentModelFlag() {
	return (useSegmentModel != 0);
}

bool RecParam::getUseCudaFlag() {
	return (useCuda != 0);
}

std::string RecParam::getWordDictFileName() {
	return wordDictFileName;
}

int RecParam::getMultiJump() {
	return multiJump;
}

std::string RecParam::getCodebookFileName() {
	return codebookFileName;
}

double RecParam::getDurWeight() {
	return durWeight;
}

int RecParam::getRecNum() {
	return recNum;
}

bool RecParam::getTriPhone(){
	return triPhone;
}

int RecParam::getFdim(){
	return fDim;
}

std::vector<RSpeechFile> RecParam::getRecFiles() {
	return fileSets;
}

bool RecParam::getBHeadNoise(){
	return m_bHeadNOise;
}

double RecParam::getDCoef(){
	return Coef;
}
