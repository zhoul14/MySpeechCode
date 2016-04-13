/******************************************
*    2006年12月18日更新
*******************************************/
// 2007，01，01日(元旦)定稿，xiaoxi 
// 2009,02,28  增加文件字节大小属性

#ifndef	_XIAOXI_WAVEFORMAT_INC_
#define	_XIAOXI_WAVEFORMAT_INC_

#include "stdio.h"
#include <windows.h>
#include <mmsystem.h>
#include <MMreg.h>

//#define	READ_ONLY_MODE	0
//#define	READ_WRITE_MODE	1
//#define	READ_FAST_MODE	2

typedef struct tagWaveFileHeader{
	long	RIFF_FOURCC;
	long	WaveFileSize;
}	WAVE_FILE_HEADER;

typedef struct tagfmtChunk{
	long	fmt_FOURCC;
	long	fmtChunkSize;
}	FMT_CHUNK;

typedef struct tagdataChunk{
	long	data_FOURCC;
	long	dataChunkSize;
}	DATA_CHUNK;



class CWaveFile{
	private:
		static long	tagRIFF_FOURCC;
		static long	tagWAVE_FOURCC;
		static long	tagfmt_FOURCC;
		static long	tagdata_FOURCC;
		WAVE_FILE_HEADER	WaveFileHeader;
		FMT_CHUNK			fmtChunk;
		DATA_CHUNK			dataChunk;
		FILE	*fp;
		CRITICAL_SECTION	WaveFileCriticalSection; 
		bool	m_bIsFileOpen;
		long	ReadWriteMode;
		long	fmtChunkOffset;
		long	dataChunkOffset;
		long	DataBufferFileOffset;		// 数据区在文件中的起始位置
		long	MaxDataByteSize;			// 数据区长度（存放数据的字节数）
		long	ReadDataBufferPtr;			// 读数据指针
		long	WriteDataBufferPtr;			// 写数据指针
		long	m_FileByteSize;

		long	SearchChunk( FILE *fp, long Offset, long Length, long FOURCC_Chunk );
		long	UpdateWaveFileHeader(long DataByteSize);

	public:
		enum {
			modeRead		= 0,
			modeReadWrite	= 1,
			modeWrite		= 2
		};
		WAVEFORMATEX	m_WaveFormatEx;

		bool	Open(const TCHAR *WaveFileName,long DefReadWriteMode);
		long	GetFileByteSize(void)	{ return m_FileByteSize; }
		void	SetWaveFormat(WAVEFORMATEX *pWaveFormatEx);
		void	SetWaveFormat(long ChannelNum, long SampleRate,long BitsPerSample);
		long	Write(char  *DataBuffer,long ByteSize);
		long	Write(short *DataBuffer,long ByteSize);		//2006,12,18增加
		long	Read(char  *Buffer,long ByteSize);
		long	Read(short *Buffer,long ByteSize);			//2006,12,18增加

		long	Seek(long Offset);
		long	SeekReadPtr(long Offset);
		long	SeekWritePtr(long Offset);

		void	Close(void);
		void	EnterCriticalSection(void);
		void	LeaveCriticalSection(void);

		long	GetChannelNum(void);
		long	GetSampleRate(void);
		long	GetBitsPerSample(void);
		long	GetBlockAlign(void);
		long	GetDataFileOffset(void);
		long	GetDataByteSize(void);
		bool	IsFileOpen(void)
		{
			return(m_bIsFileOpen);
		};

		long	GetStereoAudio( short ( *AudioBuf )[2] , long DataNum );

		CWaveFile();
		~CWaveFile();

		short m_Table_8bit_LinearPCM_2_16bitPCM[256];
		short m_Table_8bit_MuLawPCM_2_16bitPCM[256];
		short m_Table_8bit_ALawPCM_2_16bitPCM[256];

};

#endif