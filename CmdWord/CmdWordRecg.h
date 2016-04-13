#ifndef _WJQ_CMD_WORD_RECG_H_
#define _WJQ_CMD_WORD_RECG_H_

#include <vector>
#include <string>
#include "../SpeechSegmentAlgorithm/SegmentAlgorithm.h"
#include "../CommonLib/Dict/WordDict.h"
#include "../GMMCodebook/GMMCodebookSet.h"
using std::vector;
using std::string;

struct CmdRecgResult {
	vector<int> rank;
	vector<double> lh;
	vector<string> text;
};

typedef std::pair<int, SegmentResult> ResPair;


class CmdWordRecg {
private:
	static bool errorLessThan(const ResPair& m1, const ResPair& m2);

	bool useCuda;

	bool useSegmentModel;

	SegmentUtility u;

	GMMCodebookSet* set;

	vector<vector<int> > cmdset;

	vector<string> cmdLabel;

	bool* allCmdMask;

	void readCmdBuf(const char* buf, int n);

	bool isWord(char c);

public:

	CmdRecgResult cmdRecg(short* sample, int sampleNum);

	int cmdRecgIdx(short* sample, int sampleNum);

	~CmdWordRecg();

	CmdWordRecg(const char* cmdWordFile, const char* codebookFile, const char* dictFile, double durWeight, bool useCuda, bool useSegmentModel);

	string cmdIdxToLabel(int idx);

	
};


#endif