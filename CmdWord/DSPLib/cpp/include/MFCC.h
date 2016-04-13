#ifndef	_XIAOXI_CMFCC_H_
#define	_XIAOXI_CMFCC_H_

#include "CFFT.h"

class CMFCC:public CFFTanalyser
{
	private:
		float	fres;
		long	FFT_LEN,FrameLen, SubBandNum, CepstrumNum;
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
		CMFCC(
				float	sampleF,			//数据的采样率（Hz）
				long	DefFFTLen=512,		//默认的FFT长度为512点
				long	DefFrameLen=320,	//默认的帧长为320点
				long	DefSubBandNum=24,	//默认的子带的个数为24
				long	DefCepstrumNum=14	//默认的倒谱系数个数为14
			 );
		~CMFCC();
		float	Mel(int k);
		float	DoMFCC(short *inData, float *outVect);	//返回当前帧的能量

};	

#endif
