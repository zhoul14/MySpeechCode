/******************************************
*    2006年12月08日更新
*******************************************/
// 2007，01，01日(元旦)定稿，xiaoxi 
// 2009,02,28  增加文件字节大小属性
// 2009,08,21 于苏州科技城修改成支持Unicode字符串格式
//#include "stdafx.h"
#include <TCHAR.h>
#include "CWaveFormatFile.h"
#include "io.h"

long CWaveFile::tagRIFF_FOURCC = 0x46464952;
long CWaveFile::tagWAVE_FOURCC = 0x45564157;
long CWaveFile::tagfmt_FOURCC  = 0x20746D66;
long CWaveFile::tagdata_FOURCC = 0x61746164;

CWaveFile::CWaveFile()
{
	// Wave格式缓冲区清零
	memset( &m_WaveFormatEx, 0, sizeof(m_WaveFormatEx) );
	InitializeCriticalSection( &WaveFileCriticalSection );
	m_bIsFileOpen = false;
	m_FileByteSize	= 0;

	for( long i = 0 ; i < 255; i++ )
	{
		m_Table_8bit_LinearPCM_2_16bitPCM[i] = (short)( (i - 0x80)<<8 );
	}

}

CWaveFile::~CWaveFile()
{
	Close();
	DeleteCriticalSection( &WaveFileCriticalSection );
}

void	CWaveFile::EnterCriticalSection(void)
{
	::EnterCriticalSection( &WaveFileCriticalSection );
}
void	CWaveFile::LeaveCriticalSection(void)
{
	::LeaveCriticalSection( &WaveFileCriticalSection );
}


/*************************************
SearchChunk的意图是？
20051031
*************************************/
long CWaveFile::SearchChunk( FILE *fp, long Offset, long Length, long FOURCC_Chunk )
{
long	*pFOURCC,i,ReadLen;
char	ChunkBuffer[2000];


	fseek( fp , Offset , SEEK_SET );
	ReadLen = (long)fread( ChunkBuffer, sizeof(char), 2000, fp );
	for( i = 0 ; i < ReadLen-4; i++ )
	{
		pFOURCC = (long * ) ( &ChunkBuffer[i] );
		if( pFOURCC[0] == FOURCC_Chunk )
		{
			return( i + Offset );
		}
	}
	return(-1);
}

long CWaveFile::UpdateWaveFileHeader(long ByteSize)
{
long	OffsetAfterDataChunk;

	dataChunk.data_FOURCC   = tagdata_FOURCC;
	dataChunk.dataChunkSize = ByteSize;

	fmtChunk.fmt_FOURCC		= tagfmt_FOURCC;
	fmtChunk.fmtChunkSize	= sizeof(WAVEFORMATEX);

	WaveFileHeader.RIFF_FOURCC  = tagRIFF_FOURCC;
	WaveFileHeader.WaveFileSize = sizeof(tagWAVE_FOURCC)  +
								  sizeof(fmtChunk)        + fmtChunk.fmtChunkSize   +
								  sizeof( dataChunk )     + dataChunk.dataChunkSize;
	// 计算"data"数据块之后的第一个字节在Wave文件中的起始偏移量位置			
	OffsetAfterDataChunk	= sizeof(WaveFileHeader) + WaveFileHeader.WaveFileSize;
	return(OffsetAfterDataChunk);
}


bool CWaveFile::Open(const TCHAR *WaveFileName,long DefReadWriteMode )
{
	if( m_bIsFileOpen == true ) Close();
	ReadWriteMode	=  DefReadWriteMode;
	m_bIsFileOpen	= false;

	switch(ReadWriteMode)
	{
		case modeReadWrite:			//READ_WRITE_MODE:
		{	//创建写文件
			if( _tfopen_s(&fp, WaveFileName, _T("w+b")) != 0 ) return false;
			if( fp == NULL ) 
				return(false);
			else
			{
				m_bIsFileOpen	= true;
				// 设置初始的“WAVE”数据块的字节大小为0，
				// 然后设定“data”数据块在Wave格式文件中的偏移量位置
				MaxDataByteSize			= 0;		
				DataBufferFileOffset	= UpdateWaveFileHeader(MaxDataByteSize);	
				ReadDataBufferPtr		= 0;
				WriteDataBufferPtr		= 0;
				m_WaveFormatEx.wFormatTag = WAVE_FORMAT_PCM;
				m_FileByteSize			= 0;
				return(true);
			}
			break;
		}
		case modeRead:				//READ_ONLY_MODE:
		{
			if( _tfopen_s(&fp,WaveFileName,_T("rb")) != 0 ) return false;
			if(fp == NULL ) return(false);
			m_FileByteSize = ::_filelength(::_fileno(fp));

			fseek(fp,0,SEEK_SET);
			// 读入WAVE格式文件头
			fread( &WaveFileHeader , sizeof(WAVE_FILE_HEADER) , 1 , fp );
			if( WaveFileHeader.RIFF_FOURCC != tagRIFF_FOURCC )
			{
				fclose(fp);
				return(false);
			}
		
			//printf("FileSize = %ld\n",WaveFileHeader.WaveFileSize + sizeof(WaveFileHeader) );
			// 判断是否为“WAVE”格式文件
			if( SearchChunk(fp, sizeof(WAVE_FILE_HEADER), WaveFileHeader.WaveFileSize,tagWAVE_FOURCC)== -1 )
			{
				fclose(fp);
				return(false);
			}
			// 搜索格式块索引标志位置
			fmtChunkOffset = SearchChunk(fp,sizeof(WAVE_FILE_HEADER)+sizeof(tagWAVE_FOURCC),WaveFileHeader.WaveFileSize,tagfmt_FOURCC);
			if( fmtChunkOffset == -1 ) return(false);

			fseek(fp,fmtChunkOffset,SEEK_SET);
			fread(&fmtChunk,sizeof(FMT_CHUNK),1,fp);

			if( fmtChunk.fmtChunkSize > sizeof(WAVEFORMATEX) )
				fread(&m_WaveFormatEx, sizeof(WAVEFORMATEX),1,fp);
			else
				fread(&m_WaveFormatEx,fmtChunk.fmtChunkSize,1,fp);

			if( m_WaveFormatEx.wFormatTag != WAVE_FORMAT_PCM )
			{
				fclose(fp); 
				return(false); 
			}
			// 搜索数据块索引标志位置
			dataChunkOffset = fmtChunkOffset +sizeof( fmtChunk) + fmtChunk.fmtChunkSize;
			dataChunkOffset = SearchChunk(fp, dataChunkOffset, WaveFileHeader.WaveFileSize-dataChunkOffset,tagdata_FOURCC);
			if( dataChunkOffset == -1 ) return(false);

			fseek(fp,dataChunkOffset,SEEK_SET);
			fread(&dataChunk,sizeof(DATA_CHUNK),1,fp);
			m_bIsFileOpen = true;
			// 设置"data"数据块的在文件中的起始位置和大小
			DataBufferFileOffset = dataChunkOffset + sizeof( dataChunk );
			MaxDataByteSize		 = dataChunk.dataChunkSize;
			ReadDataBufferPtr	 = 0;
			WriteDataBufferPtr	 = 0;
			return(true);
			break;
		}
		default:
			m_bIsFileOpen = false;
	}
	return(false);
}

void	 CWaveFile::SetWaveFormat(WAVEFORMATEX *pWaveFormatEx)
{
	m_WaveFormatEx = *pWaveFormatEx; 
}

long CWaveFile::Write(char *DataBuffer,long ByteSize)
{
long	BytesWritten;
	//移动文件指针,写入数据
	fseek( fp, DataBufferFileOffset + WriteDataBufferPtr, SEEK_SET ); 
	BytesWritten =(long) fwrite( DataBuffer, sizeof(char), ByteSize, fp );
	WriteDataBufferPtr += BytesWritten;
	if( WriteDataBufferPtr  > MaxDataByteSize )
	{
		MaxDataByteSize = WriteDataBufferPtr;
	}
	m_FileByteSize = ftell( fp );
	return(BytesWritten);
}

long CWaveFile::Write(short *DataBuffer,long ByteSize)		//2006,12,18增加
{
long	BytesWritten;
	//移动文件指针,写入数据
	fseek( fp, DataBufferFileOffset + WriteDataBufferPtr, SEEK_SET ); 
	BytesWritten =(long) fwrite( DataBuffer, sizeof(char), ByteSize, fp );
	WriteDataBufferPtr += BytesWritten;
	if( WriteDataBufferPtr  > MaxDataByteSize )
	{
		MaxDataByteSize = WriteDataBufferPtr;
	}
	m_FileByteSize = ftell( fp );
	return(BytesWritten);
}


long CWaveFile::Read(char *Buffer,long ByteSize)
{
long	ByteRead;

	// 计算可供读取的数据字节数
	ByteRead = MaxDataByteSize - ReadDataBufferPtr;
	if( ByteRead > ByteSize )
		ByteRead = ByteSize;

	fseek( fp, DataBufferFileOffset+ReadDataBufferPtr, SEEK_SET );
	ByteRead =(long) fread(Buffer,sizeof(char),ByteRead,fp);

	ReadDataBufferPtr += ByteRead;	//移动缓冲区指针
	return( ByteRead );
}

long CWaveFile::Read(short *Buffer,long ByteSize)			//2006,12,18增加
{
long	ByteRead;

	// 计算可供读取的数据字节数
	ByteRead = MaxDataByteSize - ReadDataBufferPtr;
	if( ByteRead > ByteSize )
		ByteRead = ByteSize;

	fseek( fp, DataBufferFileOffset+ReadDataBufferPtr, SEEK_SET );
	ByteRead =(long) fread(Buffer,sizeof(char),ByteRead,fp);

	ReadDataBufferPtr += ByteRead;	//移动缓冲区指针
	return( ByteRead );
}


long CWaveFile::Seek(long Offset)
{
	ReadDataBufferPtr  =  Offset;
	WriteDataBufferPtr =  Offset;

	if( ReadWriteMode == modeRead )
	{	//在读模式下数据缓冲区指针不能超过数据的大小
		if( Offset > MaxDataByteSize )
			ReadDataBufferPtr = MaxDataByteSize;
		return( ReadDataBufferPtr );
	}
	else
	{
		return( WriteDataBufferPtr );
	}
}

long CWaveFile::SeekReadPtr(long Offset)
{
	ReadDataBufferPtr  =  Offset;
	if( ReadWriteMode == modeRead )
	{	//在读模式下数据缓冲区指针不能超过数据的大小
		if( Offset > MaxDataByteSize )
			ReadDataBufferPtr = MaxDataByteSize;
	}
	return( ReadDataBufferPtr );
}

long CWaveFile::SeekWritePtr(long Offset)
{
	WriteDataBufferPtr =  Offset;
	return( WriteDataBufferPtr );
}

void CWaveFile::Close(void)
{
	if( m_bIsFileOpen == true )	
	{
		if( ReadWriteMode == modeReadWrite )
		{
			UpdateWaveFileHeader( MaxDataByteSize );					// 设置Wave文件头内容
			fseek(  fp, 0L, SEEK_SET );
			fwrite( &WaveFileHeader, sizeof(WaveFileHeader), 1, fp );	// 写入RIFF文件头
			fwrite( &tagWAVE_FOURCC, sizeof(tagWAVE_FOURCC), 1, fp );	// 写入WAVE格式标记
			fwrite( &fmtChunk, sizeof(fmtChunk), 1, fp );				// 写入fmt结构数据
			fwrite( &m_WaveFormatEx, sizeof(char), fmtChunk.fmtChunkSize, fp );
			fwrite( &dataChunk, sizeof(dataChunk), 1, fp );				// 写入data结构数据
		}
		fclose(fp);
		m_bIsFileOpen = false;
	}
		
}


void CWaveFile::SetWaveFormat(long ChannelNum, long SampleRate,long BitsPerSample)
{
	m_WaveFormatEx.wFormatTag		= WAVE_FORMAT_PCM;
	m_WaveFormatEx.nChannels		= (WORD)ChannelNum;
	m_WaveFormatEx.nSamplesPerSec	= SampleRate;
    m_WaveFormatEx.nAvgBytesPerSec	= ChannelNum*SampleRate*BitsPerSample/8; 
    m_WaveFormatEx.nBlockAlign		= (WORD)(ChannelNum*BitsPerSample/8);     
	m_WaveFormatEx.wBitsPerSample	= (WORD)BitsPerSample;
	m_WaveFormatEx.cbSize			= 0;
}

long	CWaveFile::GetChannelNum(void)
{
	return(m_WaveFormatEx.nChannels);
}

long CWaveFile::GetSampleRate(void)
{
	return( m_WaveFormatEx.nSamplesPerSec );
}

long	CWaveFile::GetBitsPerSample(void)
{
	return(m_WaveFormatEx.wBitsPerSample);
}

long	CWaveFile::GetBlockAlign(void)
{
	return(m_WaveFormatEx.nBlockAlign);
}

long	CWaveFile::GetDataFileOffset(void)
{
	return(DataBufferFileOffset);
}

long	CWaveFile::GetDataByteSize(void)
{
	return(MaxDataByteSize);
}

//===================================================================
//返回数据音频的长度（采样点个数）
long	CWaveFile::GetStereoAudio( short (*AudioBuf)[2] ,long DataNum)
{
	if( m_bIsFileOpen == false )
	{
		for( long i = 0 ;i < DataNum; i++ )
		{
			AudioBuf[i][0] = 0;
			AudioBuf[i][1] = 0;
		}
		return 0;
	}


	switch( m_WaveFormatEx.wFormatTag )
	{
	case WAVE_FORMAT_PCM:
		if( m_WaveFormatEx.nChannels == 2 )
		{
			if( m_WaveFormatEx.wBitsPerSample == 16 )
			{	//立体声16比特线性PCM音频数据
				long ByteRead = this->Read((char *)AudioBuf,DataNum*sizeof(short)*2);
				long DataNumRead = ByteRead/(sizeof(short)*2);
				for(long i = DataNumRead; i< DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}
				return DataNumRead;		
			}
			else if( m_WaveFormatEx.wBitsPerSample == 8 )
			{
				//立体声8比特线性PCM音频数据
				long ByteRead = this->Read((char *)AudioBuf,DataNum*sizeof(char)*2);
				long DataNumRead = ByteRead/(sizeof(char)*2);
				for(long i = DataNumRead; i< DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}

				unsigned char (*p8bitPCM)[2] = (unsigned char (*)[2])AudioBuf;
				short L_16bitPCM,R_16bitPCM;  
				for( long i = DataNumRead-1; i >= 0; i-- )
				{	//8比特线性PCM转换成16比特线性PCM
					L_16bitPCM = m_Table_8bit_LinearPCM_2_16bitPCM[ p8bitPCM[i][0] ];
					R_16bitPCM = m_Table_8bit_LinearPCM_2_16bitPCM[ p8bitPCM[i][1] ];
					AudioBuf[i][0] = L_16bitPCM;
					AudioBuf[i][1] = R_16bitPCM;
				}
				return DataNumRead;		
			}
			else
			{	//不支持24bit特，32比特的线性PCM
				for( long i = 0 ;i < DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}
				return 0;
			}
		}
		else
		{
			if( m_WaveFormatEx.wBitsPerSample == 16 )
			{	//单声道16比特线性PCM音频数据
				long ByteRead = this->Read((char *)AudioBuf,DataNum*sizeof(short));
				long DataNumRead = ByteRead/sizeof(short);
				for(long i = DataNumRead; i< DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}

				short *pMonoAudio = (short *)AudioBuf;
				for( long i = DataNumRead-1; i >= 0 ; i-- )
				{
					AudioBuf[i][0] = pMonoAudio[i];
					AudioBuf[i][1] = pMonoAudio[i];
				}

				return DataNumRead;		
			}
			else if( m_WaveFormatEx.wBitsPerSample == 8 )
			{
				//立体声8比特线性PCM音频数据
				long ByteRead = this->Read((char *)AudioBuf,DataNum*sizeof(char));
				long DataNumRead = ByteRead/(sizeof(char));
				for(long i = DataNumRead; i< DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}

				unsigned char *pMono8bitPCM = (unsigned char *)AudioBuf;
				short Mono16bitPCM;  
				for( long i = DataNumRead-1; i >= 0; i-- )
				{	//8比特线性PCM转换成16比特线性PCM
					Mono16bitPCM = m_Table_8bit_LinearPCM_2_16bitPCM[ pMono8bitPCM[i] ];
					AudioBuf[i][0] = Mono16bitPCM;
					AudioBuf[i][1] = Mono16bitPCM;
				}
				return DataNumRead;		
			}
			else
			{	//不支持24bit特，32比特的线性PCM
				for( long i = 0 ;i < DataNum; i++ )
				{
					AudioBuf[i][0] = 0;
					AudioBuf[i][1] = 0;
				}
				return 0;
			}

		}
		break;
	case WAVE_FORMAT_ALAW:
		break;
	case WAVE_FORMAT_MULAW:
		break;
	}

	for( long i = 0 ;i < DataNum; i++ )
	{
			AudioBuf[i][0] = 0;
			AudioBuf[i][1] = 0;
	}
	return 0;
}