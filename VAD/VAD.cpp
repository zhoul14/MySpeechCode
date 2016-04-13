#include "VAD.h"


CVAD::CVAD(void)
{
	m_pfDoRecg = 0;
	m_pRecognizer = 0;
	m_hRecgComplete=CreateEvent(
						NULL,	
						FALSE,			//FALSE= create auto-reset event, TRUE=manual reset
						FALSE,			//Nonsignaled state
						_T("RecgCompleteEvent"));
}

CVAD::~CVAD(void)
{
	::CloseHandle(m_hRecgComplete);
}

// 声音驱动程序的回调函数
void CVAD::SoundDataCallBack(void *pCallbackInfo,void *Buffer,long BytesLength)
{
	CVAD *pVDA =(CVAD *)pCallbackInfo;
	pVDA->DoVAD((short *)Buffer,BytesLength/sizeof(short));	//将语音输入VAD检测器
}

void	CVAD::IniClass(void *pRecognizer)
{
	m_SpeechBuffer.IniClass();
	m_SpeechDetection.IniClass();

	m_pRecognizer = pRecognizer;
	//初始化声卡
	m_Recorder.m_pCallbackInfo = this;	//设置回调参数
	m_Recorder.SetWaveDataReadyCallbackFunction(SoundDataCallBack);	//设置录音回调函数
	//开始录音
	m_Recorder.Open();
	m_Recorder.Record();
}


void	CVAD::ResetVAD(void)
{
	m_SpeechBuffer.ClearSpeechBuffer();
	m_SpeechDetection.ResetDetector();
}

void	CVAD::SetRecgCallback(long (*pfDoRecg)(void *pInfo,short *SpeechBuf,long DataNum))
{
	m_pfDoRecg = pfDoRecg;
}




//返回值：-1表示尚未检测到语音
//		  -2表示输入数据发生错误
//        >= 0 表示识别结果

long	CVAD::DoVAD(short *pSpeechBuf,long DataNum)
{
long	FrameLen,SpeechStatus;
short	tmpBuffer[320];

	FrameLen = m_SpeechDetection.m_FrameLen;
	if( DataNum != FrameLen )	return -2;
	if( FrameLen > 320 )		return -2;			//调试时避免数组越界保护

//	m_SpeechBuffer.WriteSpeechToBuffer(pSpeechBuf,FrameLen);	
//	SpeechStatus = m_SpeechBuffer.WaitForSpeech(tmpBuffer,FrameLen);	//获得10mS的语音帧
//	if( SpeechStatus == FrameLen )
	{
		//将10mS语音送到语音检测器中进行端点判决
//		if( m_SpeechDetection.DetectSpeeeh(tmpBuffer)==SPEECH_DETECTED )
		if( m_SpeechDetection.DetectSpeeeh(pSpeechBuf)==SPEECH_DETECTED )
		{
//			this->m_Recorder.Close();
			//检测到语音
			short *pVADSpeechBuf	= m_SpeechDetection.m_SpeechBuffer;
			long VADDataNum			= m_SpeechDetection.m_SpeechFrameNum*m_SpeechDetection.m_FrameLen;

			long RecgReturn = -1;
			if( m_pfDoRecg != 0 )
			{
				RecgReturn =  m_pfDoRecg(m_pRecognizer,pVADSpeechBuf,VADDataNum);
				::SetEvent(m_hRecgComplete);		//通知主线程识别完成	
			}
//			this->m_Recorder.Open();
//			this->m_Recorder.Record();
			this->ResetVAD();
			return RecgReturn;
		}
		else
		{
			return -1;	//表示尚未检测到语音
		}
	}
	return -2;	//错误
}
	
void CVAD::WaitForSpeechDetected(void)
{
	WaitForSingleObject(m_hRecgComplete,INFINITE);
}
