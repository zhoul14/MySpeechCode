#ifndef WJQ_WORD_STRUCTURE_DICT_H
#define WJQ_WORD_STRUCTURE_DICT_H
#include "../commonvars.h"
#include <string>
#include <vector>
#include <map>
using namespace std;
typedef vector<pair<int, int>> LinkInfoVec;
typedef map<string, int> PyMap;
//typedef hash_map<string, int> PyHash;

struct VLinkHeader {
	int cNum;

	int offset;
};

class WordDict {
public:
//private:

	int cbnum;

	int noiseId;

	bool triPhone ; 

// 	string VClassMapFileName;
// 
// 	string CClassMapFileName;
// 
// 	string wordToCVFileName;
// 
// 	string CVLinkInfoFileName;
// 
// 	string wordToSylFileName;
// 
// 	string wordLabelFileName;
// 
// 	string cbStructureFileName;
// 
// 	string jumpTableFileName;

	vector<vector<int> > jumpTable;

	int wordToSylTable[TOTAL_WORD_NUM];

	int CStateTable[INITIAL_WORD_NUM][INITIAL_WORD_STATE_NUM];

	int	VStateTable[FINAL_WORD_NUM][FINAL_WORD_STATE_NUM];

	int VCStateTable[V_CLASS_NUM][C_CLASS_NUM][3];

	int VClassTable[FINAL_WORD_NUM];

	int CClassTable[INITIAL_WORD_NUM];

	int wordToCVTable[TOTAL_WORD_NUM][2];

	VLinkHeader VLinkHeaderTable[FINAL_WORD_NUM];

	int validCBeforeVTable[TOTAL_WORD_NUM];

	int validWordWithVTable[TOTAL_WORD_NUM];

	int CVtoWordIdTable[INITIAL_WORD_NUM][FINAL_WORD_NUM];

	string wordLabelTable[TOTAL_WORD_NUM];

	//PyHash wordLabelToIdHash;
	PyMap wordLabelToIdMap;

	vector<int> coaFlagTable;

	vector<int> cbTypeTable;

	vector<int> CVIdFromCbIdTable;

	vector<vector<int>> UsingCVWord;

	void setVCClassTable(const string& CClassMapFileName, const string& VClassMapFileName);

	void setWordToCV(const string& wordToCVFileName);

	void setVCLinkInfo(const string& CVLinkInfoFileName);

	void setWordToSylTable(const string& wordToSylFileName);

	void setWordLabelTable(const string& wordLabelFileName);

	void setCbStructure(const string& cbStructureFileName);

	void setJumpTable(const string& jumpTableFileName);

	bool checkCbStructureEntry(int cbid, char cvflag, int pos, int coa, int classid, int nclassid, int cvid/*, int bakid*/);
	
//public:

	static const int INIT_MODE_BIN_FILE = 0;
	static const int INIT_MODE_MEMORY = 1;
	static const int INIT_MODE_XML_FILE = 2;


	static const int INVALID_STATE_ID = -1;

	int getCIdFromWordId(int wordId) const;

	int getVIdFromWordId(int wordId) const;

	void getWordCbId(int wordId, int prevWordId, int nextWordId, int* res) const;

	void getDiWordCbId(int wordId, int *res)const;

	int wordToSyllable(int wordId) const {
		if (wordId <= -1 || wordId > TOTAL_WORD_NUM) {
			printf("wordid [%d] out of range in wordToSyllable\n", wordId);
			exit(-1);
		}
// 		if (wordId == -1)
// 			return -1;

		return wordToSylTable[wordId];
	}

	int getIsoCbId(int id, int pos) const;

	int getDiIsoCbId(int id, int pos) const;

	//CA:co-articulated,返回协同发音时initial0, final2, final3的码本号
	int getCACbId(int classId, int relatedClassId, int pos) const;

	//WordDict(const string& configfile);

	WordDict(const char* ptr, bool triPhone = true);

	int getCClass(int cid) const;

	int getVClass(int vid) const;

	int getNoiseCbId() const;

	//返回pair的first元素代表cid，second代表wid
	LinkInfoVec getLinkOfV(int vid) const;

	void getUsedStateIdInAns(bool* output, int* ansList, int ansNum);

	void getUsedStateIdInAnsDi(bool* output, int* ansList, int ansNum);

	string wordToText(int wordId) const;

	int textToWord(const string& py) const;

	int getTotalCbNum() const;

	bool isCoa(int cbid) const;

	vector<vector<int> > getJumpTable();

	vector<int> getJumpList(int cbid) const;

	int getCbType(int cbid) const;

	int CVtoWordId(int cid, int vid) const;

	int getCVIdFromCbId(int cbid) const;

	int getWordIdFromCVLink(int cbIdOfC, int cbIdOfV) const;

	void saveToFile(const char* filename);

	void setTriPhone(bool flag);

	vector<int> getstateUsingWord(int stateIdx);

	void makeUsingCVidWordList();
};


#endif