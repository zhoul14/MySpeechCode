#ifndef	_INCLUDE_XIAOXI_FFT_H_
#define	_INCLUDE_XIAOXI_FFT_H_

// The CFFTanalyser class is designed to implement the FFT algorithm of real or 
// complex data sequence.
//
// CFFTanalyser: The FFT length is set up by this construction function.
//				 The default FFT length is 512.
// DoFFT:		 This function implement the in place complex FFT and 
//				 invert FFT algorithm. The length of the FFT is defined by the
//				 construction function.
// DoRealFFT:	 This function implement the real FFT algorithm. The length of 
//				 real data sequence is also defined by the construction function.
//				 flag = 0	FFT
//				 flag = 1	Invert FFT
 
class CFFTanalyser {
	private:
		float	*cosTable,*sinTable;
		long	FFT_LEN,ButterFlyDepth;
	public:
		CFFTanalyser(long FFT_AnalysePointNum=512);
		~CFFTanalyser();
		void SetupSinCosTable(void);
		void DoFFT(float *fr, float *fi, short flag );	
		void DoRealFFT(float *Buf);
		long GetFFTAnalyseLen(void);
};
	
#endif
