#include"GetNewFeature.h"


GetNewFeature::GetNewFeature(short *sample,int sampleNum_in, bool d,int DefFFTLen,int DefFrameLen):CFFTanalyser(DefFFTLen)
{
	this->samples=sample;
	this->sampleNum=sampleNum_in;
	this->FFT_Len=DefFFTLen;
	this->FrameLen=DefFrameLen;
	hamWin = new float[FrameLen];		
	FFTFrame = new float[FFT_Len];
	feature_tri=new float[N_DIM];
	FrameNum = (sampleNum - FRAME_LEN + FRAME_STEP) / FRAME_STEP;
	int i;
	float a =(float)( asin(1.0)*4/(FrameLen-1) );
	d48 = d;
	if (!d)
	{
		d42 = true;
	}

	for(i=0;i<FrameLen;i++)
		hamWin[i] = 0.54f - 0.46f * (float)cos(a*i);
	featureBuf = (float(*)[DIM+N_DIM])malloc(FrameNum * (DIM+N_DIM )* sizeof(float));
	featureMBuf = (float(*)[DIM+MIN_DIM])malloc(FrameNum * (DIM+MIN_DIM )* sizeof(float));

	if (featureBuf == NULL ) {
		printf("cannot malloc memory for FeatureBuf\n");
		exit(-1);
	}

	MakeMFCC();

	if (d48)
	{
		MakeNewFeature(d48);
	}



}

GetNewFeature::~GetNewFeature()
{
	delete []samples;
	delete []featureBuf;
	delete []featureMBuf;
	delete []hamWin;
	delete []FFTFrame;
	delete []feature_tri;
}

void GetNewFeature::MakeMFCC(){
	auto featureBuf_temp = (float(*)[DIM])malloc(FrameNum * DIM * sizeof(float));
	auto featureBuf_min = (float(*)[DIM+MIN_DIM])malloc(FrameNum * (DIM+MIN_DIM) * sizeof(float));

	if (featureBuf_temp == NULL ) {
		printf("cannot malloc memory for FeatureBuf_temp\n");
		exit(-1);
	}
	get20dBEnergyGeometryAveragMfcc(samples,featureBuf_temp, FrameNum);
	if(d48)
	{
		Make45in48(featureBuf_temp);
	}
	if(d42)
	{
		Make45in42(featureBuf_temp);
	}	
	free(featureBuf_temp);
	free(featureBuf_min);
}
void GetNewFeature::MakeNewFeature(bool fDim48){
	int i,j;
	
	for(i=0;i<FrameNum;i++){
		CalFeat(i);

		//for(j=0;j<N_DIM;j++)featureBuf[i][DIM+j]=feature_tri[j];
		for(j=0;j<N_DIM;j++)featureBuf[i][DIM+j]=feature_tri[j];

		//if (fDim48)
		//{
		//	continue;
		//}	
		/*for(j=N_DIM/2;j<N_DIM;j++)
		if(i==0)
		featureBuf[i][DIM+j]=0;
		else
		featureBuf[i][DIM+j]=featureBuf[i][DIM+j-N_DIM/2]-featureBuf[i-1][DIM+j-N_DIM/2];*/
	}
}


void GetNewFeature::Make45in48(float(*featureBuf_temp)[DIM]){
	int i,j;
	for(i=0;i<FrameNum;i++)for(j=0;j<DIM;j++)
	featureBuf[i][j]=featureBuf_temp[i][j];
}
void GetNewFeature::Make45in42(float(*feature_temp)[DIM]){
	int i,j;
	for(i=0;i<FrameNum;i++)for(j=0;j<DIM + MIN_DIM;j++)
		featureMBuf[i][j]=feature_temp[i][j];
}

void GetNewFeature::CalFeat(int idx){

	int i;short *dataBegin=&samples[idx*FRAME_STEP];
	for(i=0;i<FrameLen;i++)
	{
		FFTFrame[i] = (float)(dataBegin[i]);
	}
	for( i= FrameLen-1; i > 0; i-- )	//计算语音信号差分并进行加窗
	{
		FFTFrame[i] -= FFTFrame[i-1]*PREE;
		FFTFrame[i] *= hamWin[i];
	}
	FFTFrame[0] *= (1.0f-PREE);
	FFTFrame[0] *= hamWin[0];
	for(i=FrameLen;i<FFT_Len;i++) FFTFrame[i]=0.0f;

	DoRealFFT(FFTFrame);

	int Wide_1 = SubbandBoundary / ( DEFAULT_SAMPLE_RATE / 512 );

	int Wide_2 = SubbandBoundary / ( DEFAULT_SAMPLE_RATE / 512) * 3;

	int Wide_3 = SubbandBoundary /  ( DEFAULT_SAMPLE_RATE / 512) / 2;

	float out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;

	float energy_2500 = 0.0f, energy_6000 = 0.0f, energy = 0.0f;

	//能量归一化
	/*for ( i = 3; i < 4 * Wide_1; i++)
	energy_6000 += FFTFrame[i] * FFTFrame[i];
	energy_2500 = energy_6000;*/
	//for ( i; i < 9 * Wide_1; i++)
		//energy_6000 += FFTFrame[i] * FFTFrame[i];

	for(i=2;i<128;i++)energy+=FFTFrame[i]*FFTFrame[i];

	for (int i = 0; i < N_DIM; i++)
		feature_tri[i] = 0.0f;

	//for(int j=0;j<N_DIM;j++){
	int WideList[N_DIM] = { Wide_1, Wide_1*2.1, Wide_1*3.3, Wide_1*4.6, Wide_1 * 6.0};// Wide_1*4.2, Wide_1*5.4 };
	//int WideList[N_DIM] = { Wide_1, Wide_1*2, Wide_1*3}; 
	for (int j = 0; j < N_DIM; j++){

		//if (j)
		//{		
		//	WideList[j]+=WideList[j-1];
		//}
		for( int ii = 2; ii < Wide_1*(j+1); ii++){
			
			feature_tri[j]+=FFTFrame[ii]*FFTFrame[ii];
		}
		if (j==3)
		{
			for (int ii=Wide_1*(j+1);ii < Wide_1*(j+1)+8;ii++)
			{
				feature_tri[j]+=FFTFrame[ii]*FFTFrame[ii];
			}			
		}
		feature_tri[j]=log(feature_tri[j]/energy);//log

	}
	//for ( int j = N_dim_1; j < N_DIM; j++)
	//{
	//	for ( int ii = Wide_1 * (N_dim_1 + 1); ii < j * Wide_2 ; ii++)
	//	{
	//		feature_tri[j]+=FFTFrame[ii]*FFTFrame[ii];
	//	}
	//	feature_tri[j]=log(feature_tri[j]/energy_7500);//log
	//}
	

}
float *GetNewFeature::GetFeatureBuff(){


	if (d42)
	{
		return GetFeatureMBuff();
	}
	float *features=(float*)featureBuf;

	float* fd = new float[FrameNum * (DIM+N_DIM)];
	
	for (int j = 0; j < FrameNum * (DIM+N_DIM); j++) {
	
		fd[j] = features[j];

	}

	return fd;
}
float *GetNewFeature::GetFeatureMBuff(){

	float *features=(float*)featureMBuf;

	float* fd = new float[FrameNum * (DIM+MIN_DIM)];

	for (int j = 0; j < FrameNum * (DIM+MIN_DIM); j++) {

		fd[j] = features[j];

	}

	return fd;
}