#pragma once

#include "CFFT.h"

class CTelePhoneMfcc:public CFFTanalyser
{
public:
	CTelePhoneMfcc(
		float	CutOffFreq		= 3400,		//信号带宽,16.8个子带
		float	sampleFreq		= 8000,		//数据的采样率（Hz）
		long	DefFFTLen		= 512,		//默认的FFT长度为512点
		long	DefFrameLen		= 160,		//默认的帧长为160点(20mS@8KHz)
		long	DefSubBandNum	= 17,		//默认的子带的个数为24@16KHz
		long	DefCepstrumNum	= 14		//默认的倒谱系数个数为14
	);

	~CTelePhoneMfcc(void);

private:
		long	m_MaxFFT;
		float	m_fres;				//FFT分析的频率分辨率，用于线性频率到Mel刻度的换算
		long	m_FFT_LEN,m_FrameLen,m_SubBandNum, m_DCT_DIM, m_CepstrumNum;
		float	*cosTab;			//cosine table for DCT transformation 
		float	*hamWin;			//hamming window coefficients
		float	*cepWin;			//倒谱窗加权系数
		float	*MelBandBoundary;	//记录每个子带的起始MEL刻度
		float	*SubBandWeight;		//weighting coefficients for 
									//energy of each FFT frequence component
		long	*SubBandIndex;		//mapping of the FFT frequence component to sub-band No.
		float	*SubBandEnergy;		//for accumulating the corresponding subband energy
		float	*FFTFrame,*Cepstrum;
	public:
/*	
		~CMFCC();
*/
		float	Mel(int k);
		float	DoMFCC(short *inData, float *outVect);	//返回当前帧的能量


};

