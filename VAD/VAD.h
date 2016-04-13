#pragma once
#include "CSpeechBuffer.h"
#include "CSpeechDetection.h"
#include "CRecorder.h"


class CVAD
{
public:
	CVAD(void);
	~CVAD(void);
	void	IniClass(void *pRecognizer);
	void	*m_pRecognizer;
	CRECORDER			m_Recorder;
	CSPEECH_BUFFER		m_SpeechBuffer;
	CSPEECH_DETECTION	m_SpeechDetection;
	HANDLE				m_hRecgComplete;

	long (*m_pfDoRecg)(void *pInfo,short *SpeechBuf,long DataNum);

	static void SoundDataCallBack(void *pCallbackInfo,void *Buffer,long BytesLength);
	void	ResetVAD(void);
	void	SetRecgCallback(long (*pfDoRecg)(void *pInfo,short *SpeechBuf,long DataNum));
	long	DoVAD(short *SpeechBuf,long DataNum);
	void	WaitForSpeechDetected(void);
};
