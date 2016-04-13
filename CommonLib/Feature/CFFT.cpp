//---------------------------------------------------------------------------//
// This program is originally written by Dr. Yang HaoRong.					 //	
// In memory of Dr. Yang , the real FFT algorithm is implemented			 //
// and tested by Xiao Xi after reading Yang's program.						 //
//---------------------------------------------------------------------------//
#include <math.h>

#include "CFFT.h"

typedef struct tagComplex{
	float	Real;
	float	Img;
} COMPLEX;

CFFTanalyser::CFFTanalyser(long FFT_AnalysePointNum)
{
	ButterFlyDepth = 0;
	//do until 2**ButterFlyDepth >= FFT_AnalysePointNum 
	do{
		ButterFlyDepth++;
		FFT_LEN=1L<<ButterFlyDepth;		
	}while(FFT_LEN<FFT_AnalysePointNum);
	cosTable=new float[FFT_LEN];
	sinTable=new float[FFT_LEN];
	SetupSinCosTable();
}

CFFTanalyser::~CFFTanalyser()
{
	delete []cosTable;
	delete []sinTable;
}

long CFFTanalyser::GetFFTAnalyseLen(void)
{
	return(FFT_LEN);
}

void CFFTanalyser::SetupSinCosTable(void)
{
long	i;
double	TWO_PI;
	
	TWO_PI=4.0*asin(1.0);
	for(i = 0; i <FFT_LEN; i++ )
    {
		sinTable[i] =(float)sin( TWO_PI * i/FFT_LEN );
		cosTable[i] =(float)cos( TWO_PI * i/FFT_LEN );
	}
}
/*  Fast Fourier Transform for N points.
 *	N .................................... Sample Points Number
 *  Input :
 *      float fr[] ........................ Real part of input data
 *      float fi[] ................... Imaginary part of input data
 *      short flag ........................... Flag of direction of
 *                            flag = 0 : direct; flag = 1 : inverse
 *  Output :
 *      float fr[] ....................... Real part of output data
 *      float fi[] .................. Imaginary part of output data
 */
void CFFTanalyser::DoFFT( float *fr, float *fi, short flag )
{
long	arg, cntr, p1, p2;
long	i, j, a, b, k;
float	pr, pi, t;

	if( flag != 0 )
	{
		for( i = 0; i < FFT_LEN; i++ )
		{
			fr[i] = fr[i] / FFT_LEN;
			fi[i] = fi[i] / FFT_LEN;
		}
	}

	j = 0;
	for( i = 0; i <= FFT_LEN-2; i++ ) //倒序排列输入的数据
	{
		if( i < j )
		{
			t = fr[i];
			fr[i] = fr[j];
			fr[j] = t;
			t = fi[i];
			fi[i] = fi[j];
			fi[j] = t;
		}
		k = FFT_LEN ;		//做倒序加法
		do{
			k=k>>1;
			j=j^k;			//位异或加法
		}while((j&k)==0);
	}
	a = 2;
	b = 1;
	if( flag == 0 )
	{
		for( cntr = 1; cntr <= ButterFlyDepth; ++cntr )
		{
			p1 = FFT_LEN / a;
			p2 = 0;
			for( k = 0; k <= b-1; k++ )
			{
				i = k;
				while( i < FFT_LEN )
				{
					arg = i + b;
					if( k == 0 )
					{
						pr = fr[arg];		pi = fi[arg];
					}
					else
					{
						pr = fr[arg] * cosTable[p2] + fi[arg] * sinTable[p2];
						pi = -fr[arg] * sinTable[p2] + fi[arg] * cosTable[p2];
					}
					fr[arg] = fr[i] - pr;	fi[arg] = fi[i] - pi;
					fr[i]  += pr;			fi[i]  += pi;
					i += a;
				}
				p2 += p1;
			}
			a *= 2;
			b *= 2;
		}
	}
	else
	{
		for( cntr = 1; cntr <= ButterFlyDepth; ++cntr )
		{
			p1 = FFT_LEN / a;
			p2 = 0;
			for( k = 0; k <= b-1; k++ )
			{
				i = k;
				while( i < FFT_LEN )
				{
					arg = i + b;
					if( k == 0 )
					{
						pr = fr[arg];		pi = fi[arg];
					}
					else
					{
						pr = fr[arg] * cosTable[p2] - fi[arg] * sinTable[p2];
						pi = fr[arg] * sinTable[p2] + fi[arg] * cosTable[p2];
					}
					fr[arg] = fr[i] - pr;	fi[arg] = fi[i] - pi;
					fr[i]  += pr;			fi[i]  += pi;
					i += a;
				}
				p2 += p1;
			}
			a *= 2;
			b *= 2;
		}
	}

}
//----------------------------------------------------------------------------//
//The implementation of real sequence FFT by using the complex sequence FFT
//
//	Supposed:
//		X1(k) = FFT[x1(n)]
//		X2(k) = FFT[x2(n)]
//		G(k)  = FFT[x1(n)+j*x2(n)]
//		Ger(k) is the even component of the real part of G(k)
//		Gor(k) is the odd component of the real part of G(k)
//		Gei(k) is the even component of the imaginary part of G(k)
//		Goi(k) is the odd component of the imaginary part of G(k)
//	such that we have:
//		Ger(k) = Real[X1(k)]
//		Gor(k) = -Img[X2(k)]
//		Gei(i) = Real[X2(k)]
//		Goi(i) = Img[X1(k)]
//	hence we can express X1(k) and X2(k) by :
//		X1(k)=Ger(k) + j*Goi(k)
//		X2(k)=Gei(k) - j*Gor(k)
//	
//	When x1(n)=x(2n) and x2(n)=x(2n+1),n=0,1,...N/2-1,
//  here x(n) is an N points real sequence,then we have
//		X(k)=X1(k)+X2(k)*exp(-j*2*PI*k/N)	, k=0,1,...N/2-1
//
//---------------------------------------------------------------------------//
void CFFTanalyser::DoRealFFT(float *Buf)
{
COMPLEX *pBuf;
long	arg, cntr, p1, p2;
long	i, j, a, b, k, index;
float	pr, pi, t;
float	X1R,X1I,X2R,X2I;
long	ComplexFFTLen;

	ComplexFFTLen=FFT_LEN/2;
	pBuf=(COMPLEX *)Buf;
	j = 0;
	for( i = 0; i <= ComplexFFTLen-2; i++ ) //倒序排列输入的数据
	{
		if( i < j )
		{
			t = pBuf[i].Real;
			pBuf[i].Real = pBuf[j].Real;
			pBuf[j].Real = t;
			t = pBuf[i].Img;
			pBuf[i].Img = pBuf[j].Img;
			pBuf[j].Img = t;
		}
		k = ComplexFFTLen ;		//做倒序加法
		do{
			k=k>>1;
			j=j^k;			//位异或加法
		}while((j&k)==0);
	}
	a = 2;
	b = 1;
	//做FFT_LEN/2点FFT快速算法，蝴蝶结级数比FFT_LEN点FFT少一级
	for( cntr = 1; cntr < ButterFlyDepth; ++cntr )
	{
		p1 = ComplexFFTLen / a;
		p2 = 0;
		for( k = 0; k <= b-1; k++ )
		{
			i = k;
			while( i < ComplexFFTLen )
			{
				arg = i + b;
				if( k == 0 )
				{
					pr = pBuf[arg].Real;
					pi = pBuf[arg].Img;
				}
				else
				{
					index = p2<<1;
					pr = pBuf[arg].Real * cosTable[index] + pBuf[arg].Img  * sinTable[index];
					pi = pBuf[arg].Img  * cosTable[index] - pBuf[arg].Real * sinTable[index];
				}
				pBuf[arg].Real = pBuf[i].Real - pr;
				pBuf[arg].Img  = pBuf[i].Img -  pi;
				pBuf[i].Real  += pr;		pBuf[i].Img += pi;
				i += a;
			}
			p2 += p1;
		}
		a *= 2;
		b *= 2;
	}

//	caculate the odd and even component of real and imaginary part of
//	the complex FFT
//		X1(k)=Ger(k) + j*Goi(k)
//		X2(k)=Gei(k) - j*Gor(k)
//		X(k)=x1(k)+x2(k)*exp(-j*2*PI*k/N)

	X1R = pBuf[0].Real;		X1I = 0;
	X2R = pBuf[0].Img;		X2I = 0;
	pBuf[0].Real = X1R + X2R;
	pBuf[0].Img  = 0;

	for(i=1;i<ComplexFFTLen/2;i++)	
	{	//use the symmetry property to reduce the caculation
		X1R = (pBuf[i].Real+pBuf[ComplexFFTLen-i].Real)/2;
		X1I	= (pBuf[i].Img-pBuf[ComplexFFTLen-i].Img)/2;
		X2R	= (pBuf[i].Img+pBuf[ComplexFFTLen-i].Img)/2;
		X2I	= (pBuf[ComplexFFTLen-i].Real-pBuf[i].Real)/2;

		pBuf[i].Real = X1R + X2R*cosTable[i] + X2I*sinTable[i];
		pBuf[i].Img  = X1I - X2R*sinTable[i] + X2I*cosTable[i];
// note that: 
//		cosTable[ComplexFFTLen-i]=-cosTable[i]
//		sinTable[ComplexFFTLen-i]=sinTable[i]		
//		X1R[ComplexFFTLen-i]=X1R[i]	,	X1I[ComplexFFTLen-i]=-X1I[i]	
//		X2R[ComplexFFTLen-i]=X2R[i]	,	X2I[ComplexFFTLen-i]=-X2I[i]	

		pBuf[ComplexFFTLen-i].Real =  X1R - X2R*cosTable[i] - X2I*sinTable[i];
		pBuf[ComplexFFTLen-i].Img  = -X1I - X2R*sinTable[i] + X2I*cosTable[i];
	}
// note that:
//	X1R[ComplexFFTLen/2]=pBuf[ComplexFFTLen/2].Real
//	X2R[ComplexFFTLen/2]=pBuf[ComplexFFTLen/2].Img
//	X1I=0						,	X2I=0
//	cosTable[ComplexFFTLen/2]=0	,	sinTable[ComplexFFTLen/2]=1

	pBuf[ComplexFFTLen/2].Img = - pBuf[ComplexFFTLen/2].Img;
}


