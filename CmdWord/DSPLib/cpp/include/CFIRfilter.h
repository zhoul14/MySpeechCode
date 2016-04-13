#ifndef	_XIAOXI_CFIRFILTER_H_
#define	_XIAOXI_CFIRFILTER_H_

class	CFIRFilter{
	private:
		double			*h;
		double			*CircularBuf;
		long			FirFilterLen;		//should be an odd number;
		unsigned long	CirPtr,CircularBufLen,CircularMask;
		float	FilterSampleRate,FilterLowCutoffFreq,FilterHighCutoffFreq;
		void	GetFirFilter(double *h,long FirLen,float SampleRate,float LowCutoffFreq,float HighCutoffFreq);
	public:
		CFIRFilter(
			float LowFreqCutoff  = 0,		//滤波器低端的转折频率(Hz)
			float HighFreqCutoff = 5000,	//滤波器高端的转折频率(Hz)	
			float SampleRate     = 16000,	//采样率(Hz)
			long  FirLen		 = 301		//should be an odd number
			);
		~CFIRFilter();
		void DoFirFilter(short *InBuffer,short *OutBuffer,long DataNum);
		void DoFirFilter(double *InBuffer,double *OutBuffer,long DataNum);
		long GetFirFilterLen(void);
		void ResetFilter(void);
};

#endif


