#ifndef	_XIAOXI_RECORDER_INC_
#define	_XIAOXI_RECORDER_INC_
#pragma comment( lib, "winmm.lib" )
#include "windows.h"
#include "mmsystem.h"
#include "atlstr.h"
typedef void (*PWAVE_DATA_READY_CALLBACK_FUNCTION )(void *pCallbackInfo,void *Buffer,long BytesLength);


class	CRECORDER{
	private:
		long			m_DeviceID;
		GLOBALHANDLE    *hWaveInHdr;
		LPBYTE          *lpInBuffer;
		LPWAVEHDR       *lpWaveInHdr;
		GLOBALHANDLE    *hInBuffer;

		WAVEFORMATEX	m_WaveFormatEx;
		HANDLE			m_hWaveInThread;

		HANDLE			m_hMonitorThread;
		DWORD			dwMonitorThreadId;

	public:
		CRECORDER::CRECORDER(
			long DefSampleRate		= 16000,
			long DefChannelNum		= 1,
			long DefBitNum			= 16,
			long DefBlockSampleLen	= 160,
			long DefDeviceBufferNum = 20,
			PWAVE_DATA_READY_CALLBACK_FUNCTION pWaveCallBack = NULL);
		~CRECORDER();

		void			*m_pCallbackInfo;
		long			m_DeviceBufferNum;
		char			*m_AudioBlockBuffer;
		long			*m_AudioBlockBufferLen;
		long			m_WritePtr;
		long			m_ReadPtr;
		long			m_AudioBlockNum;
		HANDLE			m_hAuidoReadyEvent;

		long			m_DeviceBlockBufferByteSize;
		bool			m_bWaveInThreadExit;
		bool			m_bMonitorThreadExit;
		HWAVEIN				hWaveIn ;		// Wave In device handle
		PWAVE_DATA_READY_CALLBACK_FUNCTION	pWaveDataReadyCallBackFunction;
		CRITICAL_SECTION	m_csDevice;
		CRITICAL_SECTION	m_csNotifyAuidoBuffer;
		CRITICAL_SECTION	m_csSetCallback;

		DWORD			dwThreadId;
		bool			m_bDeviceOpen;
		bool			m_bRecording;
		void	SetWaveDataReadyCallbackFunction(PWAVE_DATA_READY_CALLBACK_FUNCTION pCallBackFunction);
		void	SetWaveFormat(long WaveChannelNum,long WaveSampleRate,long WaveBitNum);

		static long	GetWaveInDeviceNum(void);
		static bool	GetWaveInDeviceName(long DeviceID,CString &strDeviceName);
		bool	QuerySupportedFormat(WAVEFORMATEX *pWaveFormatEx ,long WaveInID = WAVE_MAPPER );
		bool	Open(long WaveInID = WAVE_MAPPER );

		bool	Close(void);
		bool	Record(void);
		bool	Stop(void);
		bool	IsCallbackUserApplicationComplete(void);
};
#endif

