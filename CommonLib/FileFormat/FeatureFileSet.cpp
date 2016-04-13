#include "FeatureFileSet.h"
#include "../commonvars.h"
#include <stdio.h>
#include <io.h>
#include <iostream>
#include <string>

//using namespace boost;

int FeatureFileSet::fileByteNum(FILE* f) {
	if (f == NULL)
		return -1;
	return _filelength(_fileno(f));
}



FeatureFileSet::FeatureFileSet(const std::string& featureFileName, const std::string& maskFileName, const std::string& answerFileName, int featureDim) {
	this->featureDim = featureDim;

	FILE* featureFile = fopen(featureFileName.c_str(), "rb");
	if (featureFile == NULL) {
		//std::cout << (format("feature file [%s] cannot be opened") % featureFileName).str();
		printf("feature file [%s] cannot be opened\n", featureFileName.c_str());
		exit(-1);
	}
	int n = fileByteNum(featureFile);
	featureFileBuf = new char[n];
	if (n != fread(featureFileBuf, sizeof(char), n, featureFile)) {
		//std::cout << (format("error when read feature file [%s]") % featureFileName).str();
		printf("error when read feature file [%s]\n", featureFileName.c_str());
		exit(-1);
	}
	fclose(featureFile);

	FILE* maskFile = fopen(maskFileName.c_str(), "rb");
	if (maskFile == NULL) {
		//std::cout << (format("mask file [%s] cannot be opened") % maskFileName).str();
		printf("mask file [%s] cannot be opened\n", maskFileName.c_str());
		exit(-1);
	}
	n = fileByteNum(maskFile);
	maskFileBuf = new char[n];
	if (n != fread(maskFileBuf, sizeof(char), n, maskFile)) {
		//std::cout << (format("error when read mask file [%s]") % maskFileName).str();
		printf("error when read mask file [%s]\n", maskFileName.c_str());
		exit(-1);
	}
	fclose(maskFile);

	FILE* answerFile = fopen(answerFileName.c_str(), "rb");
	if (answerFile == NULL) {
		//std::cout << (format("answer file [%s] cannot be opened") % answerFileName).str();
		printf("answer file [%s] cannot be opened\n", answerFileName.c_str());
		exit(-1);
	}
	n = fileByteNum(answerFile);
	answerFileBuf = new char[n];
	if (n != fread(answerFileBuf, sizeof(char), n, answerFile)) {
		//std::cout << (format("error when read answer file [%s]") % answerFileName).str();
		printf("error when read answer file [%s]\n", answerFileName.c_str());
		exit(-1);
	}
	fclose(answerFile);
	

	AnswerIndex* ai = (AnswerIndex*)answerFileBuf;
	int speechNum1 = ai->offset / sizeof(AnswerIndex);

	FeatureIndex* fi = (FeatureIndex*)featureFileBuf;
	int speechNum2 = fi->offset / sizeof(FeatureIndex);
	if (speechNum2 != speechNum1) {
		printf("warning: idx file has %d speeches but d45 file has %d speeches, use the small one\n", speechNum2, speechNum1);
	}
	speechNum = speechNum1 < speechNum2 ? speechNum1 : speechNum2;

	SegPtList.resize(speechNum);
}

int FeatureFileSet::getSpeechNum() {
	return this->speechNum;
}

int FeatureFileSet::getFrameNumInSpeech(int speechIdx) {

	FeatureIndex* fIdx = (FeatureIndex*)(featureFileBuf + speechIdx * sizeof(FeatureIndex));
	float* speechData = (float*)(featureFileBuf + fIdx->offset);
	int frameNum = fIdx->byteSize / sizeof(float) / featureDim;

	MaskIndex* mIdx = (MaskIndex*)(maskFileBuf + speechIdx * sizeof(MaskIndex));
	int* maskData = (int*)(maskFileBuf + mIdx->offset);
	int segNum = mIdx->endpointNum / 2;

	int totalFrameNum = 0;
	for (int i = 0; i < segNum; i++) {
		int begFrame = maskData[i * 2];
		int endFrame = maskData[i * 2 + 1];

		if (endFrame >= frameNum)
			endFrame = frameNum - 1;

		totalFrameNum += endFrame - begFrame + 1;
	}
	return totalFrameNum;
}

void FeatureFileSet::getSpeechAt(int speechIdx, double* outputBuffer) {
	if (speechIdx >= speechNum) {
		//std::cout << (format("required index %d is out of the range of total speech number (%d)") % speechIdx % speechNum).str();
		printf("required index %d is out of the range of total speech number (%d)\n", speechIdx, speechNum);
		exit(-1);
	}

	FeatureIndex* fIdx = (FeatureIndex*)(featureFileBuf + speechIdx * sizeof(FeatureIndex));
	float* speechData = (float*)(featureFileBuf + fIdx->offset);
	int frameNum = fIdx->byteSize / sizeof(float) / featureDim;

	MaskIndex* mIdx = (MaskIndex*)(maskFileBuf + speechIdx * sizeof(MaskIndex));
	int* maskData = (int*)(maskFileBuf + mIdx->offset);
	int segNum = mIdx->endpointNum / 2;

	int totalFrameNum = 0;
	int cnt = 0;
	for (int i = 0; i < segNum; i++) {
		int begFrame = maskData[i * 2];
		int endFrame = maskData[i * 2 + 1];

		if (endFrame >= frameNum)
			endFrame = frameNum - 1;

		totalFrameNum += endFrame - begFrame + 1;

		for (int j = begFrame; j <= endFrame; j++) {
			for (int k = 0; k < featureDim; k++) {
				outputBuffer[cnt * featureDim + k] = speechData[j * featureDim + k];
			}
			cnt++;
		}
	}

}

int FeatureFileSet::getWordNumInSpeech(int speechIdx) {
	if (speechIdx >= speechNum) {
		//std::cout << (format("required index %d is out of the range of total speech number (%d)") % speechIdx % speechNum).str();
		printf("required index %d is out of the range of total speech number (%d)\n", speechIdx, speechNum);
		exit(-1);
	}

	AnswerIndex* aIdx = (AnswerIndex*)(answerFileBuf + speechIdx * sizeof(AnswerIndex));
	return aIdx->wordNum;
}

void FeatureFileSet::getWordListInSpeech(int speechIdx, int* outputWordList) {
	AnswerIndex* aIdx = (AnswerIndex*)(answerFileBuf + speechIdx * sizeof(AnswerIndex));
	int* start = (int*)(answerFileBuf + aIdx->offset);
	for (int i = 0; i < aIdx->wordNum; i++) {
		outputWordList[i] = start[i];
	}

	for (int i = 0; i < aIdx->wordNum; i++) {
		int t = outputWordList[i];
		if (t < 0 || t >= TOTAL_WORD_NUM) {
			//std::cout << (format("word of index [%d] has invalid id [%d]") % i % t).str();
			printf("word of index [%d] has invalid id [%d]\n", i, t);
			exit(-1);
		}
	}

	//return aIdx->wordNum;
}

FeatureFileSet::~FeatureFileSet() {
	if (featureFileBuf != NULL)
		delete [] featureFileBuf;
	if (maskFileBuf != NULL)
		delete [] maskFileBuf;
	if (answerFileBuf != NULL)
		delete [] answerFileBuf;
}

void FeatureFileSet::SaveSegmentPointToBuf(int speechIdx, std::vector<int> &res)
{
	
	FeatureIndex* fIdx = (FeatureIndex*)(featureFileBuf + speechIdx * sizeof(FeatureIndex));
	float* speechData = (float*)(featureFileBuf + fIdx->offset);
	int frameNum = fIdx->byteSize / sizeof(float) / featureDim;

	MaskIndex* mIdx = (MaskIndex*)(maskFileBuf + speechIdx * sizeof(MaskIndex));
	int* maskData = (int*)(maskFileBuf + mIdx->offset);
	int segNum = mIdx->endpointNum / 2;

	int i = 1;
	std::vector<int>SegmentPointList;
	SegmentPointList.push_back(0);
	while(i< res.size())
	{
		if (res.at(i)!=res.at(i-1))
		{
			SegmentPointList.push_back(i);
		}
		i++;
	}
	if (*SegmentPointList.rbegin()!=res.size()-1)
		SegmentPointList.push_back(res.size()-1);
	for (int i = 0; i < SegmentPointList.size(); i++)
	{
		SegmentPointList[i] += maskData[0];
		for (int j = 1; j < segNum; j++)
		{
			if(SegmentPointList[i] < maskData[j] && SegmentPointList[i] > maskData[j-1])
			{
				SegmentPointList[i] += maskData[j-1];
				break;
			}			
		}
	}
	if(speechIdx == 0)
		SegPtList[speechIdx].offset=sizeof(long)*2*speechNum;
	else
		SegPtList[speechIdx].offset=SegPtList[speechIdx-1].offset+SegPtList[speechIdx-1].SegPtNum*sizeof(int);
	SegPtList[speechIdx].SegPtNum=SegmentPointList.size();
	SegmentpoitData.push_back(SegmentPointList);
}


void FeatureFileSet:: printFeatureFileToTxt(const char * filename){
	FILE* fid=fopen(filename,"w");
	//	for (int i=0; i<speechNum; i++)

	for (int i=0; i<1; i++)
	{
		fprintf(fid, "No.%d file:\n",i);
		int fNum=getFrameNumInSpeech(i);
		double* buf=new double[featureDim * fNum];
		getSpeechAt(i,buf);
		for (int j=0; j<fNum; j++)
		{
			fprintf(fid,"No.%d frame:\n",j);
			for (int k=0; k<featureDim; k++)
			{
				fprintf(fid, "%f\t",buf[j * featureDim + k]);
			}
			fprintf(fid,"\n");
		}
		fprintf(fid,"\n");
		delete []buf;
	}
	

	fclose(fid);
}

void FeatureFileSet::PrintSegmentPointBuf(std::string filename)
{
	FILE* fid = fopen(filename.c_str(),"wb+");
	for (int i = 0; i < speechNum ;i++)
	{ 
		fwrite(&SegPtList[i].offset,sizeof(long),1,fid);
		fwrite(&SegPtList[i].SegPtNum,sizeof(long),1,fid);
	}
	for (int i = 0; i < speechNum; i++)
	{
		for (int j = 0; j<SegmentpoitData[i].size();j++)
		{		
			fwrite(&SegmentpoitData[i][j],sizeof(int),1,fid);
		}
	}
	fclose(fid);
}
