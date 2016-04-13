#include <windows.h>
#include <math.h>
#include "MFCC.h"
#include "20dBEnergyGeometryAverageMFCC45.h"

CMFCC	CMfcc(DEFAULT_SAMPLE_RATE,
			  DEFAULT_FFT_LEN,
			  DEFAULT_FRAME_LEN,
			  DEFAULT_SUB_BAND_NUM,
			  DEFAULT_CEP_COEF_NUM);

float	aa[5] = {-0.75f, -0.375f, 0.0f, 0.375f, 0.75f};

/*-----------------------------------------------------------------------———————-----*/
/*   此乃修改后的特征程序，根据语音信号的元音部分能量来计算语音的信号能量的几何平均值       */ 
/*   元音信号的门限值是最大语音帧能量的-20dB                                                */ 
/*---------------------------------------------------------------------------———————-*/
bool get20dBEnergyGeometryAveragMfcc(short *SpeecBuffer,float (*FeatureBuffer)[DIM],long FrameNum)
{
long	i,FrameCounter,FrameNo;
float	FrameEnergy,MaxEn,AveEn;

	//第 0 ~ 13 维存放倒谱，第 14 ~ 27 维存放一阶倒谱
	for( FrameNo = 0 ; FrameNo < FrameNum; FrameNo++ )
	{
		//第 0 ~ 13 维存放倒谱
		FrameEnergy = CMfcc.DoMFCC( &SpeecBuffer[FrameNo * FRAME_STEP], FeatureBuffer[FrameNo] );	
		FeatureBuffer[FrameNo][3*D] = (float)(10.0*log10(FrameEnergy));
	}

	//搜索能量最大的语音帧 ------ 2001,6,11修订
	MaxEn = -1000;
	for( FrameNo = 0; FrameNo < FrameNum; FrameNo++ )
	{
		if( FeatureBuffer[FrameNo][3*D] > MaxEn )
			MaxEn = FeatureBuffer[FrameNo][3*D];
	}
	//计算语音的平均能量	
	AveEn		 = 0.0;
	FrameCounter = 0;
	for( FrameNo = 0; FrameNo < FrameNum; FrameNo++ )
	{	//以最大能量的-20dB为元音信号的门限值计算语音的平均能量
		if( FeatureBuffer[FrameNo][3*D] > MaxEn -20 )
		{
			AveEn += FeatureBuffer[FrameNo][3*D];
			FrameCounter++;
		}
	}
	AveEn = AveEn/FrameCounter;
	// 归一化语音帧能量
	for( FrameNo =0 ; FrameNo < FrameNum; FrameNo++ )
		FeatureBuffer[FrameNo][3*D] = FeatureBuffer[FrameNo][3*D]-AveEn;

	//计算头两帧和最后两帧的一阶差分倒谱
	for( i = 0; i < D; i++ )
	{
		FeatureBuffer[0][D+i]=(float)(2.0*(aa[1]*FeatureBuffer[0][i]+aa[3]*FeatureBuffer[1][i]));		//第0帧
		FeatureBuffer[1][D+i]=(float)(2.0*(aa[1]*FeatureBuffer[0][i]+aa[3]*FeatureBuffer[2][i]));		//第1帧
		FeatureBuffer[FrameNum-2][D+i]=(float)(2.0*(aa[1]*FeatureBuffer[FrameNum-3][i]+aa[3]*FeatureBuffer[FrameNum-1][i]) );
		FeatureBuffer[FrameNum-1][D+i]=(float)(2.0*(aa[1]*FeatureBuffer[FrameNum-2][i]+aa[3]*FeatureBuffer[FrameNum-1][i]) );
	}
	//计算中间帧的一阶差分倒谱
	for( FrameNo = 2; FrameNo < FrameNum-2; FrameNo++ ) 
	{
		for( i = 0; i < D; i++ )
			FeatureBuffer[FrameNo][D+i] = 
				aa[0]*FeatureBuffer[FrameNo-2][i] +
				aa[1]*FeatureBuffer[FrameNo-1][i] +
				aa[3]*FeatureBuffer[FrameNo+1][i] +
				aa[4]*FeatureBuffer[FrameNo+2][i];
	}

	/************************************/
	//计算头两帧和最后两帧的二阶差分倒谱
	for( i = 0; i < D; i++ )
	{
		FeatureBuffer[0][2*D+i]=(float)(2.0*(aa[1]*FeatureBuffer[0][D+i]+aa[3]*FeatureBuffer[1][D+i]));
		FeatureBuffer[1][2*D+i]=(float)(2.0*(aa[1]*FeatureBuffer[0][D+i]+aa[3]*FeatureBuffer[2][D+i]));
		FeatureBuffer[FrameNum-2][2*D+i]=(float)(2.0*(aa[1]*FeatureBuffer[FrameNum-3][D+i]+aa[3]*FeatureBuffer[FrameNum-1][D+i]) );
		FeatureBuffer[FrameNum-1][2*D+i]=(float)(2.0*(aa[1]*FeatureBuffer[FrameNum-2][D+i]+aa[3]*FeatureBuffer[FrameNum-1][D+i]) );
	}
	//计算第 FrameNo 帧倒谱的二阶差分
	for( FrameNo = 2; FrameNo < FrameNum-2; FrameNo++ ) 
	{
		for( i = 0; i < D; i++ )
			FeatureBuffer[FrameNo][2*D+i] = 
				aa[0]*FeatureBuffer[FrameNo-2][D+i] +
				aa[1]*FeatureBuffer[FrameNo-1][D+i] +
				aa[3]*FeatureBuffer[FrameNo+1][D+i] +
				aa[4]*FeatureBuffer[FrameNo+2][D+i];
	}

	//计算头两帧和最后两帧的一阶能量差分
	FeatureBuffer[0][3*D+1]	= (float)(0.8*(aa[1]*FeatureBuffer[0][3*D] + aa[3]*FeatureBuffer[1][3*D]));
	FeatureBuffer[1][3*D+1]	= (float)(0.8*(aa[1]*FeatureBuffer[0][3*D] + aa[3]*FeatureBuffer[2][3*D]));
	FeatureBuffer[FrameNum-2][3*D+1] = (float)(0.8*(aa[1]*FeatureBuffer[FrameNum-3][3*D] + aa[3]*FeatureBuffer[FrameNum-1][3*D]));
	FeatureBuffer[FrameNum-1][3*D+1] = (float)(0.8*(aa[1]*FeatureBuffer[FrameNum-2][3*D] + aa[3]*FeatureBuffer[FrameNum-1][3*D]));
	//计算中间帧的一阶能量差分
	for( FrameNo = 2; FrameNo < FrameNum-2; FrameNo++ )
		FeatureBuffer[FrameNo][3*D+1] = 
			(float)(0.4*(aa[0]*FeatureBuffer[FrameNo-2][3*D]
					   + aa[1]*FeatureBuffer[FrameNo-1][3*D]
					   + aa[3]*FeatureBuffer[FrameNo+1][3*D]
					   + aa[4]*FeatureBuffer[FrameNo+2][3*D]));

	//计算头两帧和最后两帧的二阶能量差分
	FeatureBuffer[0][3*D+2] = (float)(1.5*(FeatureBuffer[1][3*D+1]-FeatureBuffer[0][3*D+1]));
	FeatureBuffer[FrameNum-1][3*D+2] = (float)(1.5*(FeatureBuffer[FrameNum-1][3*D+1] - FeatureBuffer[FrameNum-2][3*D+1]));
	//计算中间帧的二阶能量差分
	for(FrameNo=1;FrameNo<FrameNum-1;FrameNo++)
		FeatureBuffer[FrameNo][3*D+2] = (float)(1.5*(FeatureBuffer[FrameNo+1][3*D+1] - FeatureBuffer[FrameNo-1][3*D+1]));

	return TRUE;
}


