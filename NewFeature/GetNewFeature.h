#include "20dBEnergyGeometryAverageMFCC45.h"
#include<iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdio.h>
#include "CFFT.h"
using namespace std;
#define SubbandBoundary 500 
//#define N_DIM 5
#define N_DIM 5
#define MIN_DIM -3
#define N_dim_1 4
#define PREE        0.98f
class GetNewFeature:public CFFTanalyser
{
public:
	GetNewFeature(short *sample,int sampleNum_in, bool d = false, int DefFFTLen=512,int DefFrameLen=320);

	~GetNewFeature();

	bool SaveBin();

	void CalFeat(int idx);

	inline int GetFrameNum(){return this->FrameNum;}

	float* GetFeatureBuff();

	float *GetFeatureMBuff();

	bool d48Ord42(){
		return d48;
	}
private:

	short *samples;

	int sampleNum;

	float (*featureBuf)[DIM+N_DIM];

	float (*featureMBuf)[DIM+MIN_DIM];

	int FFT_Len;

	float *hamWin;

	float *FFTFrame;

	int FrameLen;

	int FrameNum;

	float *feature_tri;

	bool d48;

	bool d42;
	
	void GetNewFeature::Make45in48(float(*featureBuf_temp)[DIM]);

	void GetNewFeature::MakeMFCC();

	void GetNewFeature::MakeNewFeature(bool fDim48) ;
	
	void GetNewFeature::Make45in42(float(*feature_temp)[DIM]);

};

