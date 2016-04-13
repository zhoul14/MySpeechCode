#ifndef	_XIAOXI_CFIRFILTER_H_
#define	_XIAOXI_CFIRFILTER_H_

class	CFIRFilter{
	private:
		enum{
			FIR_FILTER_LEN = 321,
			CIR_BUFFER_LEN = 512,	//必须是2的n次方
		};
		double			m_FirFilter[FIR_FILTER_LEN];
		double			m_CircularBuf[CIR_BUFFER_LEN];
		long			FirFilterLen;		//should be an odd number;
		unsigned long	CirPtr,CircularBufLen,CircularMask;
		float	FilterSampleRate,FilterLowCutoffFreq,FilterHighCutoffFreq;
		void	GetFirFilter(double *h,long FirLen,float SampleRate,float LowCutoffFreq,float HighCutoffFreq);
	public:
		CFIRFilter();
		~CFIRFilter();
		void IniClass(
			float LowFreqCutoff  = 0,		//滤波器低端的转折频率(Hz)
			float HighFreqCutoff = 20,		//滤波器高端的转折频率(Hz)	
			float SampleRate     = 16000	//采样率(Hz)
			);
		void DoFirFilter(short *InBuffer,short *OutBuffer,long DataNum);
		void DoFirFilter(double *InBuffer,double *OutBuffer,long DataNum);
		long GetFirFilterLen(void);
		long GetDelayTime(void);
		void ResetFilter(void);
};

#endif


