#ifndef	_XIAOXI_SPEECH_BUFFER_INC_
#define	_XIAOXI_SPEECH_BUFFER_INC_
//#include "windows.h."
//#include "CFIRfilter.h"



class CSPEECH_BUFFER{
	private:
		enum{
			BUFFER_LEN = 0x10000			//默认的缓冲区是64K个采样点，注意长度必须是2的幂次方
		};
		unsigned long		WritePtr,ReadPtr,Mask,RawSpeechBufferLen;
		short				m_RawSpeechBuffer[BUFFER_LEN];
		short				m_TmpFilterBuffer[BUFFER_LEN];
		long				FilterSpeechBufferLen;
//		CFIRFilter			*pCBPSoundPreFilter;
//		CRITICAL_SECTION	BufferCriticalSection;
//		HANDLE				hBufferReady;
	public:
		CSPEECH_BUFFER();
		~CSPEECH_BUFFER();
		void	IniClass();

		 bool	WriteSpeechToBuffer(short *Buffer,long DataNum);	//if return false -> buffer is full
		bool	bRecorderCloseFlag;
//		long	ReadSpeechFromBuffer(short *Buffer,long DataNum);
		long	WaitForSpeech(short *Buffer,long DataNum);
		void	ClearSpeechBuffer(void);
};
#endif