//#include "stdafx.h"
#include "windows.h"
#include "atlstr.h"
#include <process.h>
#include "CRecorder.h"

DWORD WINAPI pWavInMonitorThread(LPVOID lpArg)
{
CRECORDER	*pRecorder = (CRECORDER *)lpArg;
long	AudioBlockNum,ReadPtr;

	HANDLE hThread = ::GetCurrentThread();
	::SetThreadPriority(hThread,THREAD_PRIORITY_TIME_CRITICAL);
	pRecorder->m_bMonitorThreadExit = false;
	while(1)
	{
		::WaitForSingleObject(pRecorder->m_hAuidoReadyEvent,INFINITE);
		if( pRecorder->m_bDeviceOpen == false ) 
			break;

		::EnterCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );
		AudioBlockNum = pRecorder->m_AudioBlockNum;
		::LeaveCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );

		for( long i = 0 ; i < AudioBlockNum ; i++ )
		{
			ReadPtr = pRecorder->m_ReadPtr;
			pRecorder->m_ReadPtr++;
			pRecorder->m_ReadPtr = pRecorder->m_ReadPtr % pRecorder->m_DeviceBufferNum;	//修改读地址
			
			::EnterCriticalSection(&(pRecorder->m_csSetCallback));
			if( pRecorder->pWaveDataReadyCallBackFunction != NULL )
			{	// 回调应用程序提供的处理函数
				pRecorder->pWaveDataReadyCallBackFunction( 
					pRecorder->m_pCallbackInfo,
					pRecorder->m_AudioBlockBuffer + ReadPtr*pRecorder->m_DeviceBlockBufferByteSize, 
					pRecorder->m_AudioBlockBufferLen[ReadPtr] 
					);
			}
			::LeaveCriticalSection(&(pRecorder->m_csSetCallback));
			
			::EnterCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );
			pRecorder->m_AudioBlockNum--;
			::LeaveCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );
		}
	}
	pRecorder->m_bMonitorThreadExit = true;
	return 0;
}


// 此线程用来接收录音设备发出的回调消息
DWORD WINAPI pWavInCallbackThread(LPVOID lpArg)
{
CRECORDER	*pRecorder = (CRECORDER *)lpArg;
LPWAVEHDR	lpWaveInHdr;	
MSG			Msg;

	pRecorder->m_bWaveInThreadExit = false;

	HANDLE hThread = ::GetCurrentThread();
	::SetThreadPriority(hThread,THREAD_PRIORITY_TIME_CRITICAL);
	
	// 对线程消息队列进行循环查询，直到接收到WM_QUIT消息后就终止线程
	// 系统在创建线程时，在默认的情况下不会创建线程的消息队列，直到在线程中第一次
	// 调用USER32或是Windows GDI函数时，才会自动创建线程的消息队列。

	while( GetMessage(&Msg,NULL,0,0) )
	{	// 第一次调用GetMessage函数时，线程消息队列被创建
		switch(Msg.message)
		{
		case MM_WIM_CLOSE:
			// pRecorder->m_bDeviceOpen = false;	//这里不能设置设备标志，因为有可能在线程响应此消息之前，
													//设备又被重新打开。
			PostQuitMessage(0);						//向线程的消息队列投送WM_QUIT消息后立刻放回
			break;
		case MM_WIM_DATA:

			::EnterCriticalSection( &(pRecorder->m_csDevice) );
				lpWaveInHdr=(LPWAVEHDR)Msg.lParam;
				//录音数据函数回调
				if(  lpWaveInHdr->dwBytesRecorded > 0  )
				{
					::EnterCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );
					if( pRecorder->m_AudioBlockNum < pRecorder->m_DeviceBufferNum )
					{	//复制数据到缓冲区中
						memcpy(	pRecorder->m_AudioBlockBuffer + pRecorder->m_WritePtr*pRecorder->m_DeviceBlockBufferByteSize,
								lpWaveInHdr->lpData,
								lpWaveInHdr->dwBytesRecorded 
							   );
						pRecorder->m_AudioBlockBufferLen[pRecorder->m_WritePtr] = lpWaveInHdr->dwBytesRecorded;
						pRecorder->m_WritePtr++;
						pRecorder->m_WritePtr = pRecorder->m_WritePtr%pRecorder->m_DeviceBufferNum;
						pRecorder->m_AudioBlockNum++;
						SetEvent(pRecorder->m_hAuidoReadyEvent);
					}
					::LeaveCriticalSection( &(pRecorder->m_csNotifyAuidoBuffer) );
				}

				if( pRecorder->m_bDeviceOpen == true )
				{
					waveInUnprepareHeader( pRecorder->hWaveIn, lpWaveInHdr, sizeof(WAVEHDR) );
					lpWaveInHdr->dwBufferLength  = pRecorder->m_DeviceBlockBufferByteSize;
					lpWaveInHdr->dwBytesRecorded = 0L ;
					// lpWaveInHdr->dwUser       = 0L ;
					lpWaveInHdr->dwFlags         = 0L ;
					lpWaveInHdr->dwLoops         = 0L ;
					lpWaveInHdr->lpNext          = NULL ;
					lpWaveInHdr->reserved        = 0L ;
					waveInPrepareHeader( pRecorder->hWaveIn , lpWaveInHdr , sizeof(WAVEHDR) );
					waveInAddBuffer( pRecorder->hWaveIn, lpWaveInHdr, sizeof(WAVEHDR) );
				}
			::LeaveCriticalSection( &(pRecorder->m_csDevice) );
			break;
		case MM_WIM_OPEN:
			pRecorder->m_bDeviceOpen = true;
			break;
		default:
			break;
		}
	}
	pRecorder->m_bWaveInThreadExit = true;
	return 0;
}


CRECORDER::CRECORDER(long DefSampleRate,long DefChannelNum,long DefBitNum,long DefBlockSampleLen,long DefDeviceBufferNum,
					 PWAVE_DATA_READY_CALLBACK_FUNCTION pDefWaveDataReadyCallBackFunction
					 )
{
long	i;

	m_DeviceID						= WAVE_MAPPER;
	m_DeviceBufferNum				= DefDeviceBufferNum;									//设备录音缓冲区的个数
	m_DeviceBlockBufferByteSize		= ((DefBitNum + 7 )/8)*DefBlockSampleLen*DefChannelNum;	//每个录音缓冲区的字节大小
	pWaveDataReadyCallBackFunction	= pDefWaveDataReadyCallBackFunction;					//录音数据回调函数

	hWaveInHdr		= new GLOBALHANDLE[m_DeviceBufferNum];
	lpInBuffer		= new LPBYTE[m_DeviceBufferNum];
	lpWaveInHdr		= new LPWAVEHDR[m_DeviceBufferNum];
	hInBuffer		= new GLOBALHANDLE[m_DeviceBufferNum];

	m_AudioBlockBuffer	= new char[m_DeviceBufferNum*m_DeviceBlockBufferByteSize]; 
	m_AudioBlockBufferLen  = new long[m_DeviceBufferNum];
	m_WritePtr		= 0;
	m_ReadPtr		= 0;
	m_AudioBlockNum	= 0;
	
	m_hAuidoReadyEvent = ::CreateEvent(
		NULL,
		FALSE,		// Auto Reset,
		FALSE,		// No Signal initially
		NULL
		);
	InitializeCriticalSection( &m_csNotifyAuidoBuffer);
	InitializeCriticalSection( &m_csSetCallback);
	InitializeCriticalSection( &m_csDevice );

	for( i = 0 ; i < m_DeviceBufferNum ; i++ )
	{
		//allocate WaveIn header
		hWaveInHdr[i]	= GlobalAlloc( GHND | GMEM_SHARE , sizeof(WAVEHDR) ) ;
		lpWaveInHdr[i]	= (LPWAVEHDR)GlobalLock(hWaveInHdr[i]) ;
		//allocate WaveIn audio buffer
		hInBuffer[i]	= GlobalAlloc(GHND| GMEM_SHARE , m_DeviceBlockBufferByteSize) ;
		lpInBuffer[i]	= (LPBYTE) GlobalLock(hInBuffer[i]) ;
		lpWaveInHdr[i]->lpData			= (char *)lpInBuffer[i];
		lpWaveInHdr[i]->dwBufferLength  = m_DeviceBlockBufferByteSize ;
	}

	SetWaveFormat(DefChannelNum,DefSampleRate,DefBitNum);

	m_bDeviceOpen = false;
	m_bRecording  = false;
	m_bWaveInThreadExit = true;

	return;
}

long	CRECORDER::GetWaveInDeviceNum(void)
{
	return(waveInGetNumDevs()); 
}
bool	CRECORDER::GetWaveInDeviceName(long DeviceID,CString &strDeviceName)
{
WAVEINCAPS	WaveInCaps;

	if( MMSYSERR_NOERROR == waveInGetDevCaps(DeviceID, &WaveInCaps,sizeof(WaveInCaps)) )
	{
		strDeviceName = WaveInCaps.szPname;
		return true;
	}
	else
		return false;
}


bool	CRECORDER::QuerySupportedFormat(WAVEFORMATEX *pWaveFormatEx,long WaveInDeviceID)
{
	if( WAVERR_BADFORMAT == waveInOpen(	NULL, 
				WaveInDeviceID, 
				pWaveFormatEx,
				(DWORD_PTR)NULL,
				(DWORD_PTR)NULL,
				WAVE_FORMAT_QUERY
				)	)
		return false;
	else
	    return true;
}


bool CRECORDER::Open(long WaveInID)
{
	if( m_bDeviceOpen == true ) return true;
	if( !QuerySupportedFormat( &m_WaveFormatEx ,WaveInID ) ) 
		return false;
	// 创建录音消息监视线程
	m_hWaveInThread = ::CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)pWavInCallbackThread,this,0,&dwThreadId);

	if( MMSYSERR_NOERROR == waveInOpen(	&hWaveIn, 
				WaveInID, 
				&m_WaveFormatEx,
				(DWORD_PTR)dwThreadId,
				(DWORD_PTR)this,
				CALLBACK_THREAD
				)	)
	{
		m_bDeviceOpen = true;
	}

	if(m_bDeviceOpen == false) return false;
	m_WritePtr		= 0;
	m_ReadPtr		= 0;
	m_AudioBlockNum	= 0;
	::ResetEvent(m_hAuidoReadyEvent);
	m_hMonitorThread = ::CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)pWavInMonitorThread,this,0,&dwMonitorThreadId);

	waveInGetID(hWaveIn,(LPUINT)(&m_DeviceID));		//读取当前录音设备的设备标识符 

	for( long i = 0 ; i < m_DeviceBufferNum ; i++ )
	{
       	lpWaveInHdr[i]->dwBytesRecorded = 0L ;
		lpWaveInHdr[i]->dwUser          = i ;
		lpWaveInHdr[i]->dwFlags         = 0L ;
		lpWaveInHdr[i]->dwLoops         = 0L ;
		lpWaveInHdr[i]->lpNext          = NULL ;
		lpWaveInHdr[i]->reserved        = 0L ;
		waveInPrepareHeader(hWaveIn,lpWaveInHdr[i],sizeof(WAVEHDR));
		waveInAddBuffer(hWaveIn, lpWaveInHdr[i], sizeof(WAVEHDR));
	}
	return true;
}

// 开始录音
bool CRECORDER::Record(void)
{
	if( m_bDeviceOpen == false ) return false;

	::EnterCriticalSection(&m_csDevice);
	waveInStart(hWaveIn);			//启动录音
	m_bRecording = true;
	::LeaveCriticalSection(&m_csDevice);
	return true;
}

// 暂停录音
bool CRECORDER::Stop(void)
{
	if( m_bDeviceOpen == false )	return false;
	::EnterCriticalSection(&m_csDevice);
	//waveInStop(hWaveIn);	// waveInStop命令只是停止录音,并不会返回尚未填满的缓冲区,
	waveInReset(hWaveIn);	// 而waveInReset命令则不但是停止录音，复位录音设备 ,
							// 清空录音设备中正在排队等待的录音缓冲区，而且还会将所有尚未
							// 填满的缓冲区全部被标志为DONE,返回应用程序
	m_bRecording = false;
	::LeaveCriticalSection(&m_csDevice);

	// waveInReset后，所有的返回的缓冲区是以消息的形式，通知录音消息监视线程。
	// 录音消息监视线程，在线程的消息队列中依次处理这些消息，并调用应用程序所提供的函数来处理数据。
	// 下面判断所有的录音数据是否都被应用程序接收处理完毕
	while ( IsCallbackUserApplicationComplete() == false )
	{	// 这时候消息回调线程还会将录音缓冲区重新加入设备
		// 等待应用程序处理完所有的已经采集的数据
		Sleep(100);
	}
	return true;
}

bool CRECORDER::Close(void)
{
long i;

	if( m_bDeviceOpen == false )	return false;

	::EnterCriticalSection( &m_csDevice );	
		m_bDeviceOpen = false;		// 由于这时m_bDeviceOpen = false,用来通知消息回调线程系统
									// 正在关闭录音设备,因此消息回调线程不会将录音缓冲区重新加入设备
		waveInReset(hWaveIn);		// 在关闭录音设备之前,录音设备中不能还有录音缓冲区
		for( i = 0 ; i < m_DeviceBufferNum ; i++ )
		{
			waveInUnprepareHeader(hWaveIn, lpWaveInHdr[i], sizeof(WAVEHDR));
			lpWaveInHdr[i]->dwFlags  = 0L; 
		}
		waveInClose(hWaveIn);	//关闭录音设备，录音设备关闭后，录音消息线程会自动销毁。
	::LeaveCriticalSection( &m_csDevice );

	while( m_bWaveInThreadExit == false )
	{
		Sleep(10);
	}

	::SetEvent(m_hAuidoReadyEvent);
	while( m_bMonitorThreadExit == false )
	{
		Sleep(10);
	}
	::CloseHandle(m_hWaveInThread);		// 关闭线程句柄
	::CloseHandle(m_hMonitorThread);
	return true;
}

CRECORDER::~CRECORDER()
{
long	i;

	Close();
	for( i = 0 ; i < m_DeviceBufferNum ; i++ )
	{
		GlobalUnlock(hWaveInHdr[i]);
		GlobalFree(hWaveInHdr[i]);
		GlobalUnlock(hInBuffer[i]);
		GlobalFree(hInBuffer[i]);
	}
	DeleteCriticalSection( &m_csDevice );
	DeleteCriticalSection( &m_csNotifyAuidoBuffer );
	DeleteCriticalSection( &m_csSetCallback);
	::CloseHandle(m_hAuidoReadyEvent);
	delete []hWaveInHdr;
	delete []lpInBuffer;
	delete []lpWaveInHdr;
	delete []hInBuffer;
	delete []m_AudioBlockBuffer;
	delete []m_AudioBlockBufferLen;
}


void CRECORDER::SetWaveDataReadyCallbackFunction(PWAVE_DATA_READY_CALLBACK_FUNCTION pCallBackFunction)
{
	::EnterCriticalSection(&m_csSetCallback);
	pWaveDataReadyCallBackFunction = pCallBackFunction;
	::LeaveCriticalSection(&m_csSetCallback);
}

// 设置录音数据格式
void	CRECORDER::SetWaveFormat(long WaveChannelNum,long WaveSampleRate,long WaveBitNum)
{
	m_WaveFormatEx.wFormatTag		= WAVE_FORMAT_PCM; 
    m_WaveFormatEx.nChannels		= (unsigned short)WaveChannelNum; 
    m_WaveFormatEx.nSamplesPerSec	= WaveSampleRate;	//sample rate in Hz
    m_WaveFormatEx.nAvgBytesPerSec	= (WaveChannelNum*WaveBitNum/8)*WaveSampleRate; 
    m_WaveFormatEx.nBlockAlign		= (WORD)(WaveChannelNum*WaveBitNum/8); 
    m_WaveFormatEx.wBitsPerSample	= (unsigned short)WaveBitNum; 
    m_WaveFormatEx.cbSize			= 0; 
}


bool	CRECORDER::IsCallbackUserApplicationComplete(void)
{
long	i;
bool	bComplete;

	::EnterCriticalSection(&m_csDevice);
	bComplete = true;
	for( i = 0 ; i < m_DeviceBufferNum ; i++ )
	{
		if( lpWaveInHdr[i]->dwFlags & WHDR_DONE )
		{
			bComplete = false;
			break;
		}
	}
	::LeaveCriticalSection(&m_csDevice);
	return bComplete;
}

//
//线程模式 
//waveInOpen(&hWaveIn,WAVE_MAPPER,&waveform,m_ThreadID,NULL,CALLBACK_THREAD)，
//我们可以继承MFC的CwinThread类，只要相应的处理线程消息即可。 
//MFC线程消息的宏为： 
//ON_THREAD_MESSAGE, 
//可以这样添加消息映射： ON_THREAD_MESSAGE(MM_WIM_CLOSE, OnMM_WIM_CLOSE) 
