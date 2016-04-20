#include "FrameWarehouse.h"
#include <windows.h>
#include <iostream>
#include <fstream>
//#include "boost/format.hpp"
//#include "boost/filesystem.hpp"

using std::string;

void FrameWarehouse::checkCbId(int cbid) const {
	if (cbid < 0 || cbid >= cbNum) {
		printf("cbid[%d] out of range in FrameWarehouse::ckeckCbId\n", cbid);
		exit(-1);
	}
}

FrameWarehouse::~FrameWarehouse() {
	delete [] data;
	delete [] frameCbId;
	delete [] totalFrameNumPerCb;
	delete [] bufferedFrameNumPerCb;
}

FrameWarehouse::FrameWarehouse(const std::string& dir, int cbNum, int fDim, int maxFNum, bool flg) {
	this->dir = dir;
	this->cbNum = cbNum;
	this->fDim = fDim;
	this->maxFNum = maxFNum;
	this->m_bSeged = flg;

	arrayPtr = 0;
	totalFrameNum = 0;

	try
	{
		data = new float[maxFNum * fDim];
		frameCbId = new int[maxFNum];
		totalFrameNumPerCb = new int[cbNum];
		bufferedFrameNumPerCb = new int[cbNum];
	}
	catch(std::bad_alloc &memExp)
	{
		printf("can not alloc such size memory%d\n",maxFNum*fDim*4);
	}
	if(data == nullptr)printf("Can't malloc %d size bytes\n", maxFNum * fDim * 4);
	
	memset(frameCbId, 0, maxFNum * sizeof(int));
	memset(totalFrameNumPerCb, 0, cbNum * sizeof(int));
	memset(bufferedFrameNumPerCb, 0, cbNum * sizeof(int));

	//using boost::format;
	char tmpName[200];
	//string tmpName;
	for (int i = 0; i < cbNum; i++) {
		//tmpName = (format("%s\\%04d.tmp") % dir % i).str();
		sprintf(tmpName, "%s\\%04d.tmp", dir.c_str(), i);
		string str(tmpName);
		tmpNames.push_back(str);
	}

	clearFrames();
}

void FrameWarehouse::clearFrames() {
// 	using namespace boost::filesystem;
// 
// 	path root(dir);
// 	if (!exists(root))
// 		create_directory(root);
// 	else {
// 		for (int i = 0; i < cbNum; i++) {
// 			path x(tmpNames.at(i));
// 			if (exists(x))
// 				remove(x);
// 		}
// 	}
	if (GetFileAttributes(dir.c_str()) == INVALID_FILE_ATTRIBUTES) {
		CreateDirectory(dir.c_str(), NULL);
	} else {
		for (int i = 0; i < cbNum; i++) {
			const char* x = tmpNames[i].c_str();
			if (GetFileAttributes(x) != INVALID_FILE_ATTRIBUTES) {
				DeleteFile(x);
			}
		}
	}
	memset(totalFrameNumPerCb, 0, cbNum * sizeof(int));
	memset(bufferedFrameNumPerCb, 0, cbNum * sizeof(int));
	arrayPtr = 0;
	totalFrameNum = 0;
}


void FrameWarehouse::flush() {
	if (arrayPtr == 0)
		return;

	for (int i = 0; i < cbNum; i++) {
		if (bufferedFrameNumPerCb[i] == 0)
			continue;
		
		FILE* fid = fopen(tmpNames.at(i).c_str(), "ab");
		if (!fid) {
			printf("cannot write tmp file %s\n", tmpNames.at(i).c_str());
			exit(-1);
		}

		
		for (int j = 0; j < arrayPtr; j++) {
			if (frameCbId[j] != i)
				continue;
			fwrite(data + j * fDim, sizeof(float), fDim, fid);
		}
		fclose(fid);
	}
	printf("flush!\n");
	arrayPtr = 0;
	memset(bufferedFrameNumPerCb, 0, cbNum * sizeof(int));
}

int FrameWarehouse::getFileSize(string filename) {
	std::ifstream in(filename, std::ios::ate|std::ios::binary);
	int size = in.tellg();
	in.close();
	return size;
}

void FrameWarehouse::pushFrame(int cbid, double* frame) {
	checkCbId(cbid);

	float* start = data + arrayPtr * fDim;

	for (int i = 0; i < fDim; i++){
		start[i] = frame[i];
	}
	frameCbId[arrayPtr] = cbid;

	arrayPtr++;

	totalFrameNumPerCb[cbid]++;
	bufferedFrameNumPerCb[cbid]++;
	totalFrameNum++;

	if (arrayPtr >= maxFNum)
		flush();



}

int FrameWarehouse::getFrameNum(int cbid) const {
	checkCbId(cbid);
	return totalFrameNumPerCb[cbid];
}

void FrameWarehouse::setSegedFlag(bool flg)
{
	m_bSeged = flg;
}

void FrameWarehouse::loadFrames(int cbid, double* buf) {
//	using boost::filesystem::path;
	checkCbId(cbid);
	flush();
	string tmpName = tmpNames.at(cbid);
	FILE* fid = fopen(tmpName.c_str(), "rb");
	//printf("fopen done\n");

	if (!fid) {
		printf("cannot open file[%s] to read in loadFrames\n", tmpName.c_str());
		perror("error code: ");
		exit(-1);
	}
	int fileSize = getFileSize(tmpName);

	/*if (m_bSeged)
	{
		totalFrameNumPerCb[cbid] = fileSize / fDim / sizeof(float);
	}*/
	int fNum = totalFrameNumPerCb[cbid];
	//path p(tmpName);
	//int fileSize = file_size(p);
	if (fileSize != fNum * fDim * sizeof(float)) {
		printf("fileSize=%d, should be %d\n", fileSize, fNum * fDim * sizeof(float));
		exit(-1);
	}

	float* fbuf = new float[fNum * fDim];
	fread(fbuf, sizeof(float), fNum * fDim, fid);
	for (int i = 0; i < fNum * fDim; i++) {
		buf[i] = fbuf[i];
	}

	fclose(fid);
	delete [] fbuf;
}

int FrameWarehouse::getBufferedFrameNum() const {
	return arrayPtr;
}

int FrameWarehouse::getTotalFrameNum() const {
	return totalFrameNum;
}

void FrameWarehouse:: printTmpToFile(int idx){
	string tmpName = tmpNames.at(idx);
	FILE* fid = fopen(tmpName.c_str(), "rb");
	if (!fid) {
		printf("cannot open file[%s] to read in loadFrames\n", tmpName.c_str());
		perror("error code: ");
		exit(-1);
	}
}