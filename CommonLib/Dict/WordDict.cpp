#include "WordDict.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
//#include <boost/property_tree/xml_parser.hpp>
//#include <boost/algorithm/string.hpp>

int WordDict::getTotalCbNum() const {
	return cbnum;
}

void WordDict::getUsedStateIdInAns(bool* flag, int* ansList, int ansNum) {

	for (int i = 0; i < cbnum; i++)
		flag[i] = false;

	int len=triPhone?STATE_NUM_IN_WORD:STATE_NUM_IN_DI;


	int *cbInfo = new int[len];
	for (int i = 0; i < ansNum; i++) {
		int prevWord = i == 0 ? NO_PREVIOUS_WORD : ansList[i - 1];
		int nextWord = i == ansNum - 1 ? NO_NEXT_WORD : ansList[i + 1];
		int word = ansList[i];

		if(triPhone)
			getWordCbId(word, prevWord, nextWord, cbInfo);
		else 
			getDiWordCbId(word, cbInfo);

		for (int j = 0; j < len; j++) {
			if (cbInfo[j] != WordDict::INVALID_STATE_ID) {
				flag[cbInfo[j]] = true;
			}
		}
	}

	delete cbInfo;

	//不论answer内容noise状态总会用到
	flag[noiseId] = true;
}

bool WordDict::checkCbStructureEntry(int cbid, char cvflag, int pos, int coa, int classid, int nclassid, int cvid/*, int bakid*/) {
	if (cbid < 0 || cbid >= cbnum) {
		return false;
	}

	if (cvflag != 'I' && cvflag != 'F' && cvflag != 'N') {
		return false;
	}

	if (cvflag == 'I') {
		if (pos < 0 || pos >= INITIAL_WORD_STATE_NUM) {
			return false;
		}
	}

	if (cvflag == 'F') {
		if (pos < 0 || pos >= FINAL_WORD_STATE_NUM) {
			return false;
		}
	}

	if (coa != 0 && coa != 1) {
		return false;
	}

	if (coa == 0 && (classid != -1 || nclassid != -1)) {
		return false;
	}

	if (coa == 1 && cvid != -1) {
		return false;
	}

	if (cvflag != 'N') {
		if (coa) {
			int cclass, vclass;
			if (cvflag == 'I') {
				cclass = classid;
				vclass = nclassid;
			} else {
				cclass = nclassid;
				vclass = classid;
			}
			if (cclass < 0 || cclass >= C_CLASS_NUM)
				return false;
			if (vclass < 0 || cclass >= V_CLASS_NUM)
				return false;
		} else {
			if (cvflag == 'I') {
				if (cvid < 0 || cvid >= INITIAL_WORD_NUM)
					return false;
			} else {
				if (cvid < 0 || cvid >= FINAL_WORD_NUM) {
					return false;
				}
			}
		}
	}

	return true;

}

int WordDict::getCbType(int cbid) const {
	if (cbid < 0 || cbid >= cbnum) {
		return INVALID_CB_TYPE;
	}

	return cbTypeTable[cbid];
}

void WordDict::setCbStructure(const string& cbStructureFileName) {
	ifstream in(cbStructureFileName);
	if (!in) {

		printf("Codebook structure file [%s] cannot be opened\n", cbStructureFileName.c_str());
		exit(-1);
	}
	in >> cbnum;
	if (cbnum <= 0) {
		printf("invalid cbnum [%d]\n", cbnum);
		exit(-1);
	}

	//bakTable.resize(cbnum, -1);
	coaFlagTable.resize(cbnum, -1);
	CVIdFromCbIdTable.resize(cbnum, -1);
	cbTypeTable.resize(cbnum, -1);


	int cbid, pos, coa, classid, nclassid, cvid/*, bakid*/;
	char cvflag;
	noiseId = -1;
	bool* cover = new bool[cbnum];
	for (int i = 0; i < cbnum; i++) {
		cover[i] = false;
	}

	for (int i = 0; i < cbnum; i++) {
		in >> cbid >> cvflag >> pos >> coa >> classid >> nclassid >> cvid;
		bool isValid = checkCbStructureEntry(cbid, cvflag, pos, coa, classid, nclassid, cvid);
		if (!isValid) {
			printf("invalid entry: cbid=%d, cvflag=%c, pos=%d, coa=%d, classid=%d, nclassid=%d, cvid=%d\n", 
				cbid, cvflag, pos, coa, classid, nclassid, cvid);
			exit(-1);
		}

		coaFlagTable.at(i) = coa;
		CVIdFromCbIdTable.at(i) = cvid;

		if (cvflag == 'N') {
			noiseId = cbid;
			cover[noiseId] = true;
			cbTypeTable.at(i) = triPhone? TAIL_NOISE:DI_TAIL_NOISE;
			continue;
		}



		if (coa) {
			int cclass, vclass;
			if (cvflag == 'I') {
				cclass = classid;
				vclass = nclassid;

			} else if (cvflag == 'F') {
				vclass = classid;
				cclass = nclassid;
			} else {
				printf("unrecognized cvflag %c\n", cvflag);
				exit(-1);
			}

			if (cvflag == 'F' && pos == 2) {
				VCStateTable[vclass][cclass][0] = cbid;
				cbTypeTable.at(i) = FINAL2_C;
			} else if (cvflag == 'F' && pos == 3) {
				VCStateTable[vclass][cclass][1] = cbid;
				cbTypeTable.at(i) = FINAL3_C;
			} else if (cvflag == 'I' && pos == 0) {
				VCStateTable[vclass][cclass][2] = cbid;
				cbTypeTable.at(i) = INITIAL0_C;
			} else {
				printf("unexpected coa entry\n");
				exit(-1);
			}
		} else {
			if (cvflag == 'I') {
				CStateTable[cvid][pos] = cbid;
				if (pos == 0) {
					cbTypeTable.at(i) = triPhone? INITIAL0 : DI_INITIAL0;
				} else if (pos == 1) {
					cbTypeTable.at(i) = triPhone? INITIAL1 : DI_INITIAL1;
				}

			} else {
				VStateTable[cvid][pos] = cbid;
				if (pos == 0) {
					cbTypeTable.at(i) = triPhone? FINAL0 : DI_FINAL0;
				} else if (pos == 1) {
					cbTypeTable.at(i) = triPhone? FINAL1 : DI_FINAL1;
				} else if (pos == 2) {
					cbTypeTable.at(i) = triPhone? FINAL2 : DI_FINAL2;
				} else if (pos == 3) {
					cbTypeTable.at(i) = triPhone? FINAL3 : DI_FINAL3;
				} else {
					printf("error in build cbTypeTable\n");
					exit(-1);
				}
			}
		}
		cover[i] = true;

	}

	in.close();
	for (int i = 0; i < cbnum; i++) {
		if (!cover[i]) {
			printf("codebook id %d is not found in structure file\n", i);
			exit(-1);
		}
	}
	delete [] cover;
}

void WordDict::setWordToCV(const string& wordToCVFileName) {
	ifstream in(wordToCVFileName);
	if (!in) {
		printf("word to CV file [%s] cannot be opened\n", wordToCVFileName.c_str());
		exit(-1);
	}
	for (int i = 0; i < TOTAL_WORD_NUM; i++) {
		in >> wordToCVTable[i][0] >> wordToCVTable[i][1];
	}
	in.close();
}

void WordDict::setVCLinkInfo(const string& CVLinkInfoFileName) {
	ifstream in(CVLinkInfoFileName, ios::binary);
	if (!in) {
		printf("CV link info file [%s] cannot be opened\n", CVLinkInfoFileName.c_str());
		exit(-1);
	}
	in.read((char*)(VLinkHeaderTable), sizeof(VLinkHeader) * TOTAL_WORD_NUM);
	in.read((char*)(validCBeforeVTable), sizeof(int) * TOTAL_WORD_NUM);
	in.read((char*)(validWordWithVTable), sizeof(int) * TOTAL_WORD_NUM);
	in.close();

	for (int i = 0; i < INITIAL_WORD_NUM; i++) {
		for (int j = 0; j < FINAL_WORD_NUM; j++) {
			CVtoWordIdTable[i][j] = -1;
		}
	}

	for (int i = 0; i < FINAL_WORD_NUM; i++) {
		LinkInfoVec info = getLinkOfV(i);
		for (int j = 0; j < info.size(); j++) {
			int cid = info.at(j).first;
			int wid = info.at(j).second;
			CVtoWordIdTable[cid][i] = wid;
		}
	}

}

int WordDict::CVtoWordId(int cid, int vid) const {
	if (cid < 0 || vid < 0 || cid >= INITIAL_WORD_NUM || vid >= FINAL_WORD_NUM) {
		printf("invalid cid/vid value, cid = %d, vid = %d\n", cid, vid);
		exit(-1);
	}
	return CVtoWordIdTable[cid][vid];
}

void WordDict::setVCClassTable(const string& CClassMapFileName, const string& VClassMapFileName) {

	ifstream CClassFile(CClassMapFileName);
	if (!CClassFile) {
		printf("C class map file [%s] cannot be opened\n", CClassMapFileName.c_str());
		exit(-1);
	}
	for (int i = 0; i < INITIAL_WORD_NUM; i++) {
		CClassFile >> CClassTable[i];
	}
	CClassFile.close();

	ifstream VClassFile(VClassMapFileName);
	if (!VClassFile) {
		printf("V class map file [%s] cannot be opened\n", VClassMapFileName.c_str());
		exit(-1);
	}
	for (int i = 0; i < FINAL_WORD_NUM; i++) {
		VClassFile >> VClassTable[i];
	}
	VClassFile.close();
}


void WordDict::setJumpTable(const string& jumpTableFileName) {
	ifstream inFile(jumpTableFileName);
	if (!inFile) {
		//cout << (boost::format() % jumpTableFileName).str();
		printf("jump table file [%s] cannot be opened", jumpTableFileName.c_str());
		exit(-1);
	}
	string line;
	while (getline(inFile, line)) {
		stringstream ss(line);
		vector<int> tvec;
		int t;
		while (ss >> t) {
			tvec.push_back(t);
		}
		jumpTable.push_back(tvec);
	}
	inFile.close();
}

void WordDict::setWordLabelTable(const string& wordLabelFileName) {
	ifstream inFile(wordLabelFileName);
	if (!inFile) {
		//cout << (boost::format("") % wordToSylFileName).str();
		printf("word label file [%s] cannot be opened", wordLabelFileName.c_str());
		exit(-1);
	}
	string t;
	for (int i = 0; i < TOTAL_WORD_NUM; i++) {
		inFile >> t;
		wordLabelTable[i] = t;
		wordLabelToIdMap[t] = i;
	}
	inFile.close();
}

void WordDict::setWordToSylTable(const string& wordToSylFileName) {
	ifstream inFile(wordToSylFileName);
	if (!inFile) {
		//cout << (boost::format() % wordToSylFileName).str();
		printf("word to syllable file [%s] cannot be opened", wordToSylFileName.c_str());
		exit(-1);
	}
	for (int i = 0; i < TOTAL_WORD_NUM; i++) {
		inFile >> wordToSylTable[i];
	}
	inFile.close();
}



vector<int> WordDict::getJumpList(int cbid) const {


	if ((cbid < 0 || cbid >= cbnum)&&(triPhone)) {
		printf("cbid [%d] out of range in getJumpList\n", cbid);
		exit(-1);
	}
	else if  ((cbid < 0 || cbid >= cbnum+1))
	{
		printf("cbid [%d] out of range in getJumpList\n", cbid);
		exit(-1);
	}
	return jumpTable.at(cbid);
}

vector<vector<int> > WordDict::getJumpTable() {
	return jumpTable;
}

int WordDict::getCClass(int cid) const {
	return CClassTable[cid];
}

int WordDict::getVClass(int vid) const {
	return VClassTable[vid];
}

int WordDict::getCIdFromWordId(int wordId) const {
	return wordToCVTable[wordId][0];
}

int WordDict::getVIdFromWordId(int wordId) const {
	return wordToCVTable[wordId][1];
}

int WordDict::getCACbId(int classId, int relatedClassId, int pos) const {
	//using boost::format;
	if (pos == INITIAL0_C) {
		if (classId < 0 || classId >= INITIAL_WORD_NUM || relatedClassId < 0 || relatedClassId >= FINAL_WORD_NUM) {
			printf("id[%d] or relatedId[%d] invalid\n", classId, relatedClassId);
			exit(-1);
		}
		return VCStateTable[relatedClassId][classId][2];
	} else if (pos == FINAL2_C) {
		if (classId < 0 || classId >= FINAL_WORD_NUM || relatedClassId < 0 || relatedClassId >= INITIAL_WORD_NUM) {
			printf("id[%d] or relatedId[%d] invalid\n", classId, relatedClassId);
			exit(-1);
		}
		return VCStateTable[classId][relatedClassId][0];
	} else if (pos == FINAL3_C) {
		if (classId < 0 || classId >= FINAL_WORD_NUM || relatedClassId < 0 || relatedClassId >= INITIAL_WORD_NUM) {
			printf("id[%d] or relatedId[%d] invalid\n", classId, relatedClassId);
			exit(-1);
		}
		return VCStateTable[classId][relatedClassId][1];
	} else {
		printf("pos[%d] is invalid\n", pos);
		exit(-1);
	}
	return -1;
}

int WordDict::getIsoCbId(int id, int pos) const {
	if (pos == INITIAL0 || pos == INITIAL1) {
		if (id < 0 || id >= INITIAL_WORD_NUM) {
			printf("asked cid[%d] out of range\n", id);
			exit(-1);
		}
	} else if (pos == FINAL0 || pos == FINAL1 || pos == FINAL2 || pos == FINAL3) {
		if (id < 0 || id >= FINAL_WORD_NUM) {
			printf("asked vid[%d] out of range\n", id);
			exit(-1);
		}
	} else {
		printf("pos [%d] is invalid for id [%d]\n", pos, id);
		exit(-1);
	}

	switch (pos) {
	case INITIAL0:
		return CStateTable[id][0];
	case INITIAL1:
		return CStateTable[id][1];
	case FINAL0:
		return VStateTable[id][0];
	case FINAL1:
		return VStateTable[id][1];
	case FINAL2:
		return VStateTable[id][2];
	case FINAL3:
		return VStateTable[id][3];
	default:
		return -1;	//不会执行到此处
	}
}

int WordDict::getDiIsoCbId(int id, int pos) const {
	if (pos == DI_INITIAL0 || pos == DI_INITIAL1) {
		if (id < 0 || id >= INITIAL_WORD_NUM) {
			printf("asked cid[%d] out of range\n", id);
			exit(-1);
		}
	} else if (pos == DI_FINAL0 || pos == DI_FINAL1 || pos == DI_FINAL2 || pos == DI_FINAL3) {
		if (id < 0 || id >= FINAL_WORD_NUM) {
			printf("asked vid[%d] out of range\n", id);
			exit(-1);
		}
	} else {
		printf("pos [%d] is invalid for id [%d]\n", pos, id);
		exit(-1);
	}

	switch (pos) {
	case DI_INITIAL0:
		return CStateTable[id][0];
	case DI_INITIAL1:
		return CStateTable[id][1];
	case DI_FINAL0:
		return VStateTable[id][0];
	case DI_FINAL1:
		return VStateTable[id][1];
	case DI_FINAL2:
		return VStateTable[id][2];
	case DI_FINAL3:
		return VStateTable[id][3];
	default:
		return -1;	//不会执行到此处
	}
}


void WordDict::getWordCbId(int wordId, int prevWordId, int nextWordId, int* res) const {

	int cid = wordToCVTable[wordId][0];
	int vid = wordToCVTable[wordId][1];

	res[INITIAL0] = CStateTable[cid][0];
	res[INITIAL1] = CStateTable[cid][1];
	res[FINAL0] = VStateTable[vid][0];
	res[FINAL1] = VStateTable[vid][1];
	res[FINAL2] = VStateTable[vid][2];
	res[FINAL3] = VStateTable[vid][3];

	int CClass = CClassTable[cid];
	int VClass = VClassTable[vid];

	if (prevWordId >= 0) {
		int prevVId = getVIdFromWordId(prevWordId);
		int prevVClass = VClassTable[prevVId];
		res[INITIAL0_C] = VCStateTable[prevVClass][CClass][2];
	} else {	//若是第一字
		res[INITIAL0_C] = INVALID_STATE_ID;
	}

	if (nextWordId >= 0) {
		int nextCId = getCIdFromWordId(nextWordId);
		int nextCClass = CClassTable[nextCId];
		res[FINAL2_C] = VCStateTable[VClass][nextCClass][0];
		res[FINAL3_C] = VCStateTable[VClass][nextCClass][1];
	} else {	//若是最后一字
		res[FINAL2_C] = INVALID_STATE_ID;
		res[FINAL3_C] = INVALID_STATE_ID;
	}

}

void WordDict::getDiWordCbId(int wordId, int* res) const {

	int cid = wordToCVTable[wordId][0];
	int vid = wordToCVTable[wordId][1];

	res[DI_INITIAL0] = CStateTable[cid][0];
	res[DI_INITIAL1] = CStateTable[cid][1];
	res[DI_FINAL0] = VStateTable[vid][0];
	res[DI_FINAL1] = VStateTable[vid][1];
	res[DI_FINAL2] = VStateTable[vid][2];
	res[DI_FINAL3] = VStateTable[vid][3];

	int CClass = CClassTable[cid];
	int VClass = VClassTable[vid];

}


int WordDict::getNoiseCbId() const {
	return noiseId;
}

LinkInfoVec WordDict::getLinkOfV(int vid) const {
	LinkInfoVec vec;
	if (vid < 0 || vid >= FINAL_WORD_NUM) {
		//cout << (boost::format("") % vid).str();
		printf("vid [%d] out of range\n", vid);
		exit(-1);
	}
	int cNum = VLinkHeaderTable[vid].cNum;
	int offset = VLinkHeaderTable[vid].offset;
	for (int i = 0; i < cNum; i++) {
		int cid = validCBeforeVTable[offset + i];
		int wid = validWordWithVTable[offset + i];
		vec.push_back(make_pair(cid, wid));
	}
	return vec;
}

string WordDict::wordToText(int wordId) const {
	if (wordId < 0 || wordId >= TOTAL_WORD_NUM) {
		if (wordId == -1) {	//for debug
			string noise("null");
			return noise;
		}

		printf("word id [%d] is invalid when querying label\n", wordId);
		exit(-1);
	}
	return wordLabelTable[wordId];
}

int WordDict::textToWord(const string& py) const {
	auto idx = wordLabelToIdMap.find(py);
	if (idx == wordLabelToIdMap.end()) {
		printf("invalid pinyin input to dict [%s]\n", py.c_str());
		exit(-1);
	}
	return idx->second;
}

// int WordDict::getBakId(int cbid) const {
// 	if (cbid < 0 || cbid >= cbnum) {
// 		printf("cbid [%d] is invalid in getIsoEquivalent()\n", cbid);
// 		exit(-1);
// 	}
// 
// 	return bakTable.at(cbid);
// }

bool WordDict::isCoa(int cbid) const {
	if (cbid < 0 || cbid >= cbnum) {
		printf("cbid [%d] is invalid in getIsoEquivalent()\n", cbid);
		exit(-1);
	}
	return coaFlagTable.at(cbid) == 1;
}

int WordDict::getCVIdFromCbId(int cbid) const {
	if (cbid < 0 || cbid >= cbnum) {
		printf("cbid [%d] is invalid in getCVIdFromCbId\n", cbid);
		exit(-1);
	}
	return CVIdFromCbIdTable.at(cbid);
}

int WordDict::getWordIdFromCVLink(int cbIdOfC, int cbIdOfV) const {
	int cid = getCVIdFromCbId(cbIdOfC);
	int vid = getCVIdFromCbId(cbIdOfV);
	if (cid == -1 || vid == -1) {
		printf("invalid link, cbid = (%d, %d)\n", cbIdOfC, cbIdOfV);
		exit(-1);
	}
	int wid = CVtoWordId(cid, vid);
	if (wid == -1) {
		printf("invalid wid in getWordIdFromCVLink, cbid = (%d, %d)\n", cbIdOfC, cbIdOfV);
		exit(-1);
	}
	return wid;
}

void WordDict::saveToFile(const char* filename) {
	FILE* fid = fopen(filename, "wb");
	if (!fid) {
		printf("cannot open file [%s]\n", filename);
		exit(-1);
	}
	fwrite(&cbnum, sizeof(int), 1, fid);
	fwrite(&noiseId, sizeof(int), 1, fid);

	int jumpTableLineNum = jumpTable.size();
	fwrite(&jumpTableLineNum, sizeof(int), 1, fid);
	for (int i = 0; i < jumpTableLineNum; i++) {
		int T = jumpTable[i].size();
		fwrite(&T, sizeof(int), 1, fid);
		for (int j = 0; j < T; j++) {
			int t = jumpTable[i][j];
			fwrite(&t, sizeof(int), 1, fid);
		}
	}

	fwrite(wordToSylTable, sizeof(int), TOTAL_WORD_NUM, fid);
	for (int i = 0; i < INITIAL_WORD_NUM; i++) {
		fwrite(CStateTable[i], sizeof(int), INITIAL_WORD_STATE_NUM, fid);
	}
	for (int i = 0; i < FINAL_WORD_NUM; i++) {
		fwrite(VStateTable[i], sizeof(int), FINAL_WORD_STATE_NUM, fid);
	}
	for (int i = 0; i < V_CLASS_NUM; i++) {
		for (int j = 0; j < C_CLASS_NUM; j++) {
			fwrite(VCStateTable[i][j], sizeof(int), 3, fid);
		}
	}

	fwrite(CClassTable, sizeof(int), INITIAL_WORD_NUM, fid);
	fwrite(VClassTable, sizeof(int), FINAL_WORD_NUM, fid);
	for (int i = 0; i < TOTAL_WORD_NUM; i++) {
		fwrite(wordToCVTable[i], sizeof(int), 2, fid);
	}
	fwrite(VLinkHeaderTable, sizeof(VLinkHeader), FINAL_WORD_NUM, fid);
	fwrite(validCBeforeVTable, sizeof(int), TOTAL_WORD_NUM, fid);
	fwrite(validWordWithVTable, sizeof(int), TOTAL_WORD_NUM, fid);
	for (int i = 0; i < INITIAL_WORD_NUM; i++) {
		fwrite(CVtoWordIdTable[i], sizeof(int), FINAL_WORD_NUM, fid);
	}

	for (int i = 0; i < TOTAL_WORD_NUM; i++) {
		int slen = wordLabelTable[i].size();
		fwrite(&slen, sizeof(int), 1, fid);
		fwrite(wordLabelTable[i].c_str(), sizeof(char), slen, fid);
	}

	int wToIHashLen = wordLabelToIdMap.size();
	fwrite(&wToIHashLen, sizeof(int), 1, fid);
	for (auto i = wordLabelToIdMap.begin(); i != wordLabelToIdMap.end(); i++) {
		string key = i->first;
		int value = i->second;
		int klen = key.size();
		fwrite(&klen, sizeof(int), 1, fid);
		fwrite(key.c_str(), sizeof(char), klen, fid);
		fwrite(&value, sizeof(int), 1, fid);
	}

	for (int i = 0; i < cbnum; i++) {
		int t = coaFlagTable[i];
		fwrite(&t, sizeof(int), 1, fid);
	}

	for (int i = 0; i < cbnum; i++) {
		int t = cbTypeTable[i];
		fwrite(&t, sizeof(int), 1, fid);
	}

	for (int i = 0; i < cbnum; i++) {
		int t = CVIdFromCbIdTable[i];
		fwrite(&t, sizeof(int), 1, fid);
	}

	fclose(fid);
}

WordDict::WordDict(const char* configFile,bool tri ) {
	ifstream infile(configFile);
	if (!infile) {
		printf("cannot open worddict config file %s\n", configFile);
		exit(-1);
	}
	std::unordered_map<std::string, std::string> config;
	std::string pname, pvalue;
	while (infile) {
		infile >> pname >> pvalue;
		std::transform(pname.begin(), pname.end(), pname.begin(), ::toupper);
		config[pname] = pvalue;
	}
	infile.close();

	// 		string configfile(ptr); 
	// 		using boost::property_tree::ptree;
	// 		using boost::algorithm::trim;
	// 
	// 		ptree pt;
	// 		read_xml(configfile, pt);
	// 
	// 		string VClassMapFileName = pt.get<string>("dictfile.vclass");
	// 		trim(VClassMapFileName);
	// 
	// 		string CClassMapFileName = pt.get<string>("dictfile.cclass");
	// 		trim(CClassMapFileName);
	// 
	// 		string wordToCVFileName = pt.get<string>("dictfile.wordtocv");
	// 		trim(wordToCVFileName);
	// 
	// 		string cbStructureFileName = pt.get<string>("dictfile.cbstruct");
	// 		trim(cbStructureFileName);
	// 
	// 		string CVLinkInfoFileName = pt.get<string>("dictfile.vclinkinfo");
	// 		trim(CVLinkInfoFileName);
	// 
	// 		string wordToSylFileName = pt.get<string>("dictfile.wordtosyl");
	// 		trim(wordToSylFileName);
	// 
	// 		string wordLabelFileName = pt.get<string>("dictfile.wordlabel");
	// 		trim(wordLabelFileName);
	// 
	// 		string jumpTableFileName = pt.get<string>("dictfile.jumptable");
	// 		trim(jumpTableFileName);

	triPhone=tri;
	setVCClassTable(config["CCLASS"], config["VCLASS"]);
	setCbStructure(config["CBSTRUCT"]);
	setWordToCV(config["WORDTOCV"]);
	setVCLinkInfo(config["VCLINKINFO"]);
	setWordToSylTable(config["WORDTOSYL"]);
	setWordLabelTable(config["WORDLABEL"]);
	setJumpTable(config["JUMPTABLE"]);

}

void WordDict::setTriPhone(bool flag){
	triPhone = flag;
}

vector<int> WordDict::getstateUsingWord(int stateIdx){

	int len=triPhone ? STATE_NUM_IN_WORD : STATE_NUM_IN_DI;//先单音子 占坑。

	int cvId;
	if (stateIdx < TOTAL_MONO_STATE_NUM - 1)
	{
		if (stateIdx < INITIAL_WORD_NUM * 2)
		{
			cvId = stateIdx%INITIAL_WORD_NUM;
		}
		else
		{
			cvId = (stateIdx - 2 *INITIAL_WORD_NUM)%FINAL_WORD_NUM + INITIAL_WORD_NUM;
		}
	}
	return UsingCVWord[cvId];
}

void WordDict::makeUsingCVidWordList()
{
	UsingCVWord.resize(INITIAL_WORD_NUM + FINAL_WORD_NUM);
	for (int i = 0; i != INITIAL_WORD_NUM + FINAL_WORD_NUM; i++)
	{

		if (i< INITIAL_WORD_NUM)
		{
			for (int j = 0; j != FINAL_WORD_NUM; j++)
			{
				int idx = CVtoWordIdTable[i][j];
				if (idx != -1)
				{
					UsingCVWord[i].push_back(idx);
				}
			}
		}
		else
		{
			for (int j = 0; j != INITIAL_WORD_NUM; j++)
			{
				int idx = CVtoWordIdTable[j][i - INITIAL_WORD_NUM];
				if (idx != -1)
				{
					UsingCVWord[i].push_back(idx);
				}
			}
		}
	}
}