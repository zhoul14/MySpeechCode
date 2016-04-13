#ifndef WJQ_FRAME_WAREHOUSE_H
#define WJQ_FRAME_WAREHOUSE_H

#include <string>
#include <vector>
//#include "../../CommonLib/SpeechFrame.h"

class FrameWarehouse {
	bool m_bSeged;

	int fDim;

	int cbNum;

	int maxFNum;

	float* data;

	int arrayPtr;

	int totalFrameNum;

	int* frameCbId;

	int* totalFrameNumPerCb;

	int* bufferedFrameNumPerCb;

	std::string dir;

	int getFileSize(std::string filename);

	std::vector<std::string> tmpNames;

	void checkCbId(int cbid) const;

public:
	FrameWarehouse(const std::string& dir, int cbNum, int fDim, int maxFNum, bool flg = false);

	~FrameWarehouse();

	void clearFrames();

	void pushFrame(int cidx, double* frame);

	void flush();

	void setSegedFlag(bool flg);

	int getFrameNum(int cbid) const;

	void loadFrames(int cbid, double* buf);

	int getBufferedFrameNum() const;

	int getTotalFrameNum() const;

	void printTmpToFile(int idx);
};

#endif