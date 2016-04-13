#include <stdio.h>
#include <conio.h>
#include "CSpeechDetection.h"



CSPEECH_DETECTION::CSPEECH_DETECTION( )
{

}


void	CSPEECH_DETECTION::IniClass()
{
	m_LowpassFilter.IniClass();

	m_FrameLen							= DEF_FRAME_LEN;
	m_SpeechFrameCircularBufferFrameNum	= DEF_SPEECH_FRAME_CIRCULAR_BUFFER_NUM;
	m_FrameEnergyCircularBufferFrameNum	= DEF_FRAME_ENERGY_CIRCULAR_BUFFER_NUM;
	m_MaxDetectedSpeechFrameNum			= DEF_MAX_DETECT_SPEECH_FRAME_NUM;

	if( m_MaxDetectedSpeechFrameNum < REWINDED_FRAME_NUM + REVERSE_SEARCH_FRAME + MIN_SPEECH_DURATION + POST_EXTENED_FRAME_NUM )
		m_MaxDetectedSpeechFrameNum = REWINDED_FRAME_NUM + REVERSE_SEARCH_FRAME + MIN_SPEECH_DURATION + POST_EXTENED_FRAME_NUM;

	m_SpeechCircularBufferLen		= m_SpeechFrameCircularBufferFrameNum*m_FrameLen;

	ResetDetector();

}


void CSPEECH_DETECTION::ResetDetector(void)
{
	m_LowpassFilter.ResetFilter();
	m_PreSpeechData							= 0;
	m_SpeechCircularBufferWritePtr			= 0;
	m_SpeechDetectedWritePtr				= 0;
	m_FrameEnergyCircularWritePtr			= 0;
	m_SpeechFrameNumCachedInCircualrBuffer	= 0;
	m_EnergyFrameNumCachedInCircualrBuffer	= 0;
	m_SpeechFrameNum						= 0;
	m_SkipFrame								= 0;
	m_MinEn.dB								= 200;
	m_MinEn.Duration						= 0;
	m_SpeechStatus							= WAIT_SPEECH;

	for( long i = 0 ; i < m_SpeechCircularBufferLen; i++ )
	{
		m_SpeechCircularBuffer[i] = 0;
	}

	for( long i = 0 ; i < m_FrameEnergyCircularBufferFrameNum; i++ )
	{
		m_FrameEnergyCircularBuf[i] = -1000;
	}

}

CSPEECH_DETECTION::~CSPEECH_DETECTION()
{
}

//语音检测是根据差分能量的变化进行判决的
long CSPEECH_DETECTION::DetectSpeeeh(short *Buffer)
{
long	i,j,tmpCirPtr,FrameNumRewinded,BackwardFrameNum;
double	Sum,tmp;

	if( m_SkipFrame < 2 )
	{
		//在对语音的能量进行滤波时采用的是线性相位滤波器，FIR的长度是2*m_FrameLen+1，滤波器输出会产生一帧的延迟
		//需要两帧的数据（20mS）才能填满滤波器中的缓冲区
		m_SkipFrame++;
		for( i = m_FrameLen-1 ; i> 0; i-- )
		{	//计算差分能量
			m_SpeechEnBuffer[i] = (double)(Buffer[i]-Buffer[i-1]);
			m_SpeechEnBuffer[i] = m_SpeechEnBuffer[i]*m_SpeechEnBuffer[i];
		}
		m_SpeechEnBuffer[0] = (double)(Buffer[0]-m_PreSpeechData);
		m_SpeechEnBuffer[0] = m_SpeechEnBuffer[0]*m_SpeechEnBuffer[0];
		m_PreSpeechData = Buffer[m_FrameLen-1]; 
		m_LowpassFilter.DoFirFilter(m_SpeechEnBuffer,m_LowpassSpeechEnBuffer,m_FrameLen);

		for( i = 0; i< m_FrameLen; i++ )
		{	//将语音数据添加到缓冲区中
			m_SpeechCircularBuffer[m_SpeechCircularBufferWritePtr++] = Buffer[i];
		}
		
		if( m_SpeechCircularBufferWritePtr >= m_SpeechCircularBufferLen ) m_SpeechCircularBufferWritePtr -=m_SpeechCircularBufferLen;
		m_SpeechFrameNumCachedInCircualrBuffer++;												//缓冲区中缓存的帧数加1
		if( m_SpeechFrameNumCachedInCircualrBuffer > m_SpeechFrameCircularBufferFrameNum )
			m_SpeechFrameNumCachedInCircualrBuffer = m_SpeechFrameCircularBufferFrameNum;		//数据缓冲区满


		return(WAIT_SPEECH);
	}

	for( i = 0; i< m_FrameLen; i++ )
	{	//将语音数据添加到缓冲区中
		m_SpeechCircularBuffer[m_SpeechCircularBufferWritePtr++] = Buffer[i];
	}

	for( i = m_FrameLen-1 ; i> 0; i-- )
	{	//计算差分能量
		m_SpeechEnBuffer[i] = (double)(Buffer[i]-Buffer[i-1]);
		m_SpeechEnBuffer[i] = m_SpeechEnBuffer[i]*m_SpeechEnBuffer[i];
	}
	m_SpeechEnBuffer[0] = (double)(Buffer[0]-m_PreSpeechData);
	m_SpeechEnBuffer[0] = m_SpeechEnBuffer[0]*m_SpeechEnBuffer[0];
	m_PreSpeechData = Buffer[m_FrameLen-1]; 

	if( m_SpeechCircularBufferWritePtr >= m_SpeechCircularBufferLen )
	{
		m_SpeechCircularBufferWritePtr -=m_SpeechCircularBufferLen;
	}

	m_SpeechFrameNumCachedInCircualrBuffer++;												//缓冲区中缓存的帧数加1
	if( m_SpeechFrameNumCachedInCircualrBuffer > m_SpeechFrameCircularBufferFrameNum )
		m_SpeechFrameNumCachedInCircualrBuffer = m_SpeechFrameCircularBufferFrameNum;		//数据缓冲区满
	
	m_LowpassFilter.DoFirFilter(m_SpeechEnBuffer,m_LowpassSpeechEnBuffer,m_FrameLen);
	//计算当前帧的平均对数能量
	Sum = 0;
	for( i = 0; i< m_FrameLen; i++ )
	{
		tmp = m_LowpassSpeechEnBuffer[i];
		if( tmp < 0.01 ) tmp =0.01;
		Sum += 10*log10(tmp);
	}
	Sum = Sum/m_FrameLen;
	m_FrameEnergyCircularBuf[m_FrameEnergyCircularWritePtr] = Sum;	//当前帧的平均能量

	//搜索缓冲区中的最小能量
	m_MinEn.Duration++;
	if( m_MinEn.Duration < m_FrameEnergyCircularBufferFrameNum ) 
	{	//最小能量值未超过其有效期
		if( Sum < m_MinEn.dB )
		{	m_MinEn.dB		= Sum;
			m_MinEn.Position	= m_FrameEnergyCircularWritePtr;
			m_MinEn.Duration	= 0;
		}
	}
	else
	{	//最小能量值超过了有效期，全搜索寻找最小的背景噪声能量
		m_MinEn.dB = 200;
		for( i = 0 ; i < m_FrameEnergyCircularBufferFrameNum; i++ )
		{
			if( m_FrameEnergyCircularBuf[i] < m_MinEn.dB )
			{
				m_MinEn.dB		= m_FrameEnergyCircularBuf[i];
				m_MinEn.Position	= i;
			}
		}
		// 注意：这里的m_MinEn.Duration的值本来应该是根据m_MinEn.Position与当
		//       前的m_FrameEnergyCircularWritePtr差值计算出来。但是为了简化算法，此
		//       处是直接设置m_MinEn.Duration = 0。这样下一次的全搜索更新只能
		//       是在m_SpeechFrameCircularBufferNum帧语音以后，减少了计算量。这种算
		//       法不影响最小能量的实时更新，当语音电平增加时，又具有一定的
		//       的后效性（延迟m_SpeechFrameCircularBufferNum帧更新）。

		m_MinEn.Duration = 0;
	}

	m_EnergyFrameNumCachedInCircualrBuffer++;		//能量帧数加 1 
	if( m_EnergyFrameNumCachedInCircualrBuffer > m_FrameEnergyCircularBufferFrameNum )
		m_EnergyFrameNumCachedInCircualrBuffer = m_FrameEnergyCircularBufferFrameNum;		//能量数据缓冲区满

	m_FrameEnergyCircularWritePtr++;	//循环指针加1
	if( m_FrameEnergyCircularWritePtr >= m_FrameEnergyCircularBufferFrameNum ) m_FrameEnergyCircularWritePtr -= m_FrameEnergyCircularBufferFrameNum;

	switch(m_SpeechStatus)
	{
	case WAIT_SPEECH:
		if( Sum > m_MinEn.dB + THRESH_HOLD_VOWEL_DETECTED )			//语音能量超过元音检测的门限值
		{
			//回溯语音信号的起点位置
			BackwardFrameNum = REVERSE_SEARCH_FRAME;
			if( BackwardFrameNum > m_EnergyFrameNumCachedInCircualrBuffer )
				BackwardFrameNum = m_EnergyFrameNumCachedInCircualrBuffer;

			FrameNumRewinded = 1;		//低通滤波器群时延为1帧
			tmpCirPtr = m_FrameEnergyCircularWritePtr;
			for( i = 0; i < BackwardFrameNum; i++ )
			{
				FrameNumRewinded++;
				tmpCirPtr--;
				if( tmpCirPtr < 0 ) tmpCirPtr += m_FrameEnergyCircularBufferFrameNum;
				if( m_FrameEnergyCircularBuf[tmpCirPtr] < m_MinEn.dB + THRESH_HOLD_CONSONANT_DETECTED )	//回溯截至的辅音门限
					break;
			}
			FrameNumRewinded += REWINDED_FRAME_NUM;
			if( FrameNumRewinded > m_SpeechFrameNumCachedInCircualrBuffer ) 
			{
				FrameNumRewinded = m_SpeechFrameNumCachedInCircualrBuffer;
			}
			
			tmpCirPtr = m_SpeechCircularBufferWritePtr-m_FrameLen*FrameNumRewinded;
			if( tmpCirPtr < 0 ) tmpCirPtr += m_SpeechCircularBufferLen;

			//根据检测到的起点，复制语音数据到缓冲区中
			m_SpeechDetectedWritePtr = 0;
			for( i = 0; i < FrameNumRewinded; i++ )
			{
				for( j = 0; j < m_FrameLen; j++ )
				{
					m_SpeechBuffer[m_SpeechDetectedWritePtr++] = m_SpeechCircularBuffer[tmpCirPtr++];
				}
				if( tmpCirPtr >= m_SpeechCircularBufferLen ) tmpCirPtr -= m_SpeechCircularBufferLen;
			}
			m_SpeechFrameNum		= FrameNumRewinded;
			m_VowelSpeechFrameNum	= 0;
			m_SilenceInterval		= 0;
			m_SpeechStatus			= SPEECH_START;
		}
		break;
	case SPEECH_START:
		if( m_SpeechFrameNum < m_MaxDetectedSpeechFrameNum )
		{
			for( i = 0; i< m_FrameLen; i++ )
			{
				m_SpeechBuffer[m_SpeechDetectedWritePtr++] = Buffer[i];
			}
			m_SpeechFrameNum++;
			m_VowelSpeechFrameNum++;
		}
		// 异常语音检测处理
		if( m_SpeechFrameNum >= m_MaxDetectedSpeechFrameNum )
		{
			m_SpeechFrameNum  = m_SpeechFrameNum - m_SilenceInterval;
			if( m_SpeechFrameNum > MIN_SPEECH_DURATION )
				m_SpeechStatus = SPEECH_DETECTED;		//只有语音长度大于门限值时，才认为是正常的语音
			else
				m_SpeechStatus = WAIT_SPEECH;			//语音长度小于门限值时，认为是干扰信号
			break;
		}

		if( Sum < m_MinEn.dB + THRESH_HOLD_VOWEL_TO_SILENCE )
		{
			//当元音超过3帧时认为是静音的开始，否则认为是噪声的干扰
			if( m_VowelSpeechFrameNum > 3 )
			{
				m_SpeechStatus		= SPEECH_VOWEL_TO_SILENCE;	//可能是连续语音的字间隙
				m_SilenceInterval	= 0;
			}
			else
			{	//噪声干扰，重新检测语音的起始点
				m_SpeechStatus = WAIT_SPEECH;
				//printf("Vowel too short\n");
			}
		}
		break;
	case SPEECH_VOWEL_TO_SILENCE:
		if( m_SpeechFrameNum < m_MaxDetectedSpeechFrameNum )
		{
			for( i = 0 ; i < m_FrameLen; i++ )
			{
				m_SpeechBuffer[m_SpeechDetectedWritePtr++] = Buffer[i];
			}
			m_SpeechFrameNum++;
		}
		// 异常语音检测处理
		if( m_SpeechFrameNum >= m_MaxDetectedSpeechFrameNum )
		{
			m_SpeechFrameNum  = m_SpeechFrameNum - m_SilenceInterval;
			if( m_SpeechFrameNum > MIN_SPEECH_DURATION )
				m_SpeechStatus = SPEECH_DETECTED;		//只有语音长度大于门限值时，才认为是正常的语音
			else
				m_SpeechStatus = WAIT_SPEECH;			//语音长度小于门限值时，认为是干扰信号
			break;
		}

		if( Sum < m_MinEn.dB + THRESH_HOLD_VOWEL_TO_SILENCE )		//能量小于25dB时有可能为静音
		{
			m_SilenceInterval++;
			if( m_SilenceInterval > SILENCE_INTERVAL )
			{
				m_SpeechFrameNum -= SILENCE_INTERVAL;
				if( m_SpeechFrameNum > MIN_SPEECH_DURATION )
				{
					m_SpeechFrameNum += POST_EXTENED_FRAME_NUM;		//语音结束门限到达后附加的语音帧数
					m_SpeechStatus = SPEECH_DETECTED;				//只有语音长度大于门限值时，才认为是正常的语音
				}
				else
					m_SpeechStatus = WAIT_SPEECH;					//语音长度小于门限值时，认为是干扰信号
			}
		}
		else
		{
			m_SilenceInterval		= 0;
		//	m_VowelSpeechFrameNum = 0;					//此语句屏蔽掉后，放松了语音检测的条件
			m_SpeechStatus		= SPEECH_START;
		}
		break;
	}
	printf("\r[%7.2f%7.2f]",m_MinEn.dB,Sum);
	return(m_SpeechStatus);
}
