#include <stdio.h>
#include <windows.h>
//#include "CFIRfilter.h"
#include "CSpeechBuffer.h"

//DefRawSpeechBufferLen是预设的数据缓冲区长度
//DefFilterSpeechBufferLen是预设的滤波器输出数据缓冲区长度
//CSPEECH_BUFFER类对每次送入到语音缓冲区中的数据都需要进行FIR预滤波处理，然后再进行分析处理，
//因此每次送入缓冲区的数据长度不能超过DefFilterSpeechBufferLen定义的缓冲区长度

CSPEECH_BUFFER::CSPEECH_BUFFER()
{
}

CSPEECH_BUFFER::~CSPEECH_BUFFER()
{
//	delete pCBPSoundPreFilter;

//	DeleteCriticalSection(&BufferCriticalSection);
//	CloseHandle(hBufferReady);
	return;
}


void	CSPEECH_BUFFER::IniClass()
{
	bRecorderCloseFlag		= false;
	RawSpeechBufferLen		= BUFFER_LEN;
	FilterSpeechBufferLen	= BUFFER_LEN;

//	InitializeCriticalSection(&BufferCriticalSection);

	//设置电话带通滤波器，8KHz采样率，300Hz--3300Hz带宽，301点FIR滤波器
//	pCBPSoundPreFilter= new CFIRFilter(100,3300,16000,301);
//	hBufferReady=CreateEvent(NULL,	//lpEventAttributes
//				TRUE,				//bManualReset 
//				FALSE,				//bInitialState
//				NULL);				//lpName   

	WritePtr	= 0;
	ReadPtr		= 0;
	Mask		= RawSpeechBufferLen-1;

	return;

}



bool CSPEECH_BUFFER::WriteSpeechToBuffer(short *Buffer,long DataNum)
{
long	i,EmptyBufferLen;
bool	flag;

	if( DataNum > FilterSpeechBufferLen )	// FilterSpeechBufferLen 为程序中预先设置的滤波器输出
		return(false);						// 的缓冲区pTmpFilterBuffer的长度。
											// 原始数据输入后都要进行预滤波处理
	
//	EnterCriticalSection(&BufferCriticalSection);
	
	//用带通滤波器滤波
//	pCBPSoundPreFilter->DoFirFilter((short *)Buffer,pTmpFilterBuffer,DataNum);
	memcpy(m_TmpFilterBuffer,Buffer,sizeof(short)*DataNum);	//pTmpFilterBuffer中存放的是预滤波输出结果

	//计算空余的缓冲区长度
	if( WritePtr >= ReadPtr )											//WritePtr指向第一个空白的缓冲区
		EmptyBufferLen = RawSpeechBufferLen-1 - ( WritePtr - ReadPtr );	//ReadPtr指向第一个可用的数据缓冲区
	else
		EmptyBufferLen = ReadPtr - WritePtr-1;

	if( EmptyBufferLen >= DataNum )
	{
		for(i=0;i<DataNum;i++)
		{
			m_RawSpeechBuffer[WritePtr++]=m_TmpFilterBuffer[i];
			WritePtr=WritePtr&Mask;
		}
		flag = true;
	}
	else
	{
		for( i = 0;i< EmptyBufferLen; i++ )
		{
			m_RawSpeechBuffer[WritePtr++] = m_TmpFilterBuffer[i];
			WritePtr=WritePtr&Mask;
		}
		flag = false;
	}

//	SetEvent(hBufferReady);
//	LeaveCriticalSection(&BufferCriticalSection);
	return(flag);
}
/*
long CSPEECH_BUFFER::ReadSpeechFromBuffer(short *Buffer,long DataNum)
{
long	DataReadyNum,i;

	EnterCriticalSection(&BufferCriticalSection);
	
	if( WritePtr == ReadPtr )
	{
		LeaveCriticalSection(&BufferCriticalSection);
		return(0);
	}
	if( WritePtr > ReadPtr )
	{
		DataReadyNum = WritePtr- ReadPtr;
	}
	else
	{
		DataReadyNum=RawSpeechBufferLen-ReadPtr+WritePtr;
	}
	if( DataReadyNum < DataNum )
	{
		LeaveCriticalSection(&BufferCriticalSection);
		return(0);
	}
	
	for(i=0;i<DataNum;i++)
	{
		Buffer[i]=pRawSpeechBuffer[ReadPtr++];
		ReadPtr=ReadPtr&Mask;
	}
	
	LeaveCriticalSection(&BufferCriticalSection);
	
	return(DataReadyNum);
}
*/

//等待获得DataNum个语音数据
long CSPEECH_BUFFER::WaitForSpeech(short *Buffer,long DataNum)
{
bool	bReadyFlag;
long	DataReadyNum,i;

	bReadyFlag = false;
	while( bReadyFlag == false )
	{
		bReadyFlag = true;	
//		EnterCriticalSection(&BufferCriticalSection);
		if( WritePtr == ReadPtr )
			bReadyFlag = false;
		else
		{
			if( WritePtr > ReadPtr )
				DataReadyNum = WritePtr- ReadPtr;
			else
				DataReadyNum=RawSpeechBufferLen-ReadPtr+WritePtr;

			if( DataReadyNum < DataNum )
				bReadyFlag = false;		//没有足够多的数据
		}

		if( bReadyFlag == true )
		{
			for(i=0;i<DataNum;i++)
			{
				Buffer[i]=m_RawSpeechBuffer[ReadPtr++];		//读出语音数据
				ReadPtr=ReadPtr&Mask;
			}
		}
//		LeaveCriticalSection(&BufferCriticalSection);

		if( bReadyFlag == true )
			return(DataReadyNum);
		else
		{
//			ResetEvent(hBufferReady);
//			if( WaitForSingleObject(hBufferReady,1000)==WAIT_TIMEOUT )
			{
				if( bRecorderCloseFlag == true )
					return(-2);
				else
					return(-1);
			}
		}
	}
	return(-1);
}

void	CSPEECH_BUFFER::ClearSpeechBuffer(void)
{
//	EnterCriticalSection(&BufferCriticalSection);
	WritePtr	= 0;
	ReadPtr		= 0;
//	LeaveCriticalSection(&BufferCriticalSection);
}