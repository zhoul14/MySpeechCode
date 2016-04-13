#ifndef	_XIAOXI_CSPEECH_DETECTION_H_
#define	_XIAOXI_CSPEECH_DETECTION_H_

#include	<math.h>
#include "CFIRfilter.h"

typedef struct tagSPEECH_EN{
	double	dB;
	long	Position;
	long	Duration;
} SPEECH_EN;

#define	WAIT_SPEECH				0
#define	SPEECH_START			1
#define SPEECH_VOWEL_TO_SILENCE	2
#define SPEECH_DETECTED			3


class	CSPEECH_DETECTION{
	private:
		enum{
			THRESH_HOLD_CONSONANT_DETECTED   = 9,							//辅音音检测门限
			THRESH_HOLD_VOWEL_DETECTED   = 25,								//元音检测门限
//			THRESH_HOLD_VOWEL_TO_SILENCE = THRESH_HOLD_VOWEL_DETECTED-3,	//元音结束门限
			THRESH_HOLD_VOWEL_TO_SILENCE = THRESH_HOLD_VOWEL_DETECTED-6,	//元音结束门限

			REVERSE_SEARCH_FRAME	= 20,		//从元音位置开始回溯，最多回溯150毫秒（即假设最长辅音为0.15秒）
			SILENCE_INTERVAL		= 50,		//字间隙最大为0.5秒
			MIN_SPEECH_DURATION		= 20,		//最短语音长度为0.2秒
			POST_EXTENED_FRAME_NUM	= 30,		//语音结束门限到达之后附加的静音语音帧数
			REWINDED_FRAME_NUM		= 15,		//语音起始门限之前附加的静音预留帧数

		};
		enum{
			DEF_FRAME_LEN	= 160,						//语音检测的分辨率，每次输入的点数（10mS）
			DEF_SAMPLE_RATE = 16000,					//语音数据采样率
			DEF_SPEECH_FRAME_CIRCULAR_BUFFER_NUM = 200,	//语音循环缓冲区的帧数（2秒）
			DEF_FRAME_ENERGY_CIRCULAR_BUFFER_NUM = 400, //语音帧能量的循环缓冲区帧数（4秒），用于检测背景噪声和回溯语音的起始点
			DEF_MAX_DETECT_SPEECH_FRAME_NUM		 = 2000, //所能检测的最大的语音帧数（10秒）
		};
		CFIRFilter	m_LowpassFilter;
		long	m_FrameEnergyCircularBufferFrameNum;
		long	m_SpeechFrameCircularBufferFrameNum;
		long	m_MaxDetectedSpeechFrameNum;

		long	m_SpeechCircularBufferLen;									//语音循环缓冲区的总长度
		long	m_SpeechFrameNumCachedInCircualrBuffer;						//记录缓存在循环缓冲区中语音的帧数
		long	m_EnergyFrameNumCachedInCircualrBuffer;						//记录缓存在循环缓冲区中能量的帧数

		long	m_SpeechCircularBufferWritePtr,m_FrameEnergyCircularWritePtr;	//循环缓冲区的写指针
		long	m_SpeechDetectedWritePtr;

		long	m_SkipFrame,m_VowelSpeechFrameNum;
		long	m_SpeechStatus,m_SilenceInterval;
		short	m_PreSpeechData;
		SPEECH_EN	m_MinEn;

		short	m_SpeechCircularBuffer[DEF_SPEECH_FRAME_CIRCULAR_BUFFER_NUM*DEF_FRAME_LEN];
		double	m_FrameEnergyCircularBuf[DEF_FRAME_ENERGY_CIRCULAR_BUFFER_NUM];	//用来缓存语音帧能量的循环缓冲区
		double	m_SpeechEnBuffer[DEF_FRAME_LEN];
		double	m_LowpassSpeechEnBuffer[DEF_FRAME_LEN];
public:
		short	m_SpeechBuffer[DEF_MAX_DETECT_SPEECH_FRAME_NUM*DEF_FRAME_LEN];
		long	m_FrameLen,m_SpeechFrameNum;

		CSPEECH_DETECTION();
		~CSPEECH_DETECTION( );
		void	IniClass();
		long DetectSpeeeh(short *Buffer);
		void ResetDetector(void);
};

#endif
