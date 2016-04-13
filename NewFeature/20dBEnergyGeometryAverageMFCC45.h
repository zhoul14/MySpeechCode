#ifndef _XIAOXI_MFCC46_H
#define	_XIAOXI_MFCC46_H

#define DIM     45                  /* feature vector dimemsion */
#define D       14

#define	FRAME_STEP	160
#define	FRAME_LEN	320

#define	DEFAULT_SAMPLE_RATE		16000		// 采样率
#define	DEFAULT_FFT_LEN			512			// FFT长度
#define	DEFAULT_FRAME_LEN		FRAME_LEN	// 语音的帧长
#define	DEFAULT_SUB_BAND_NUM	24			// 子带个数
#define	DEFAULT_CEP_COEF_NUM	14			// 倒谱系数个数

bool get20dBEnergyGeometryAveragMfcc(short *sentData,float (*ftrVect)[DIM],long FrameNum);


#endif



