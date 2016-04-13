#include <stdio.h>
#include <math.h>
#include "..\include\CMatrix.h"
#include "..\include\CViterbi.h"
#include "..\include\CCodeBook.h"


CVITERBI_CODEBOOK::CVITERBI_CODEBOOK()
{
long	ModelNo,i;

	pViterbiCodeBook = new VITERBI_CODEBOOK[VITERBI_CODEBOOK_NUM];
	pTmpCodeBook	 = new VITERBI_CODEBOOK[VITERBI_CODEBOOK_NUM];

	pFeaToCodeBookProb = new float[VITERBI_CODEBOOK_NUM];
	ModelStateToCodeBookNoMap = new long[VITERBI_MODEL_NUM][VITERBI_STATE_NUM];
	Log2PI =(float)log( asin(1.0)*4 );

	if( SILENCE_STATE_NUM == 2 )
	{
		for(ModelNo=0;ModelNo<VITERBI_MODEL_NUM;ModelNo++)
		{
			ModelStateToCodeBookNoMap[ModelNo][0]=VITERBI_CODEBOOK_NUM-2;
			ModelStateToCodeBookNoMap[ModelNo][VITERBI_STATE_NUM-1]=VITERBI_CODEBOOK_NUM-1;
			for(i=0;i<VITERBI_STATE_NUM-SILENCE_STATE_NUM;i++)
			{
				ModelStateToCodeBookNoMap[ModelNo][i+1]=CoreModelStateToCodeBookNoMap[ModelNo][i];
			}
		}
	}

}

CVITERBI_CODEBOOK::~CVITERBI_CODEBOOK()
{
	delete []pViterbiCodeBook;
	delete []pTmpCodeBook;
	delete []pFeaToCodeBookProb;
	delete []ModelStateToCodeBookNoMap;
}

VITERBI_CODEBOOK * CVITERBI_CODEBOOK::GetCodeBookBuffer(void)
{
	return(pViterbiCodeBook);
}

void CVITERBI_CODEBOOK::SetCodeBook(VITERBI_CODEBOOK *pCodeBook,long CodeBookNo)
{
	pViterbiCodeBook[CodeBookNo]=pCodeBook[0];
}

void CVITERBI_CODEBOOK::SetAllCodeBook(VITERBI_CODEBOOK *pCodeBook)
{
long	CodeBookNo;

	for(CodeBookNo = 0;CodeBookNo<VITERBI_CODEBOOK_NUM; CodeBookNo++)
	{
		pViterbiCodeBook[CodeBookNo]=pCodeBook[CodeBookNo];
	}
}
void CVITERBI_CODEBOOK::CalculateFeaToAllCodeBookDist(VITERBI_FEA ViterbiFea)
{
long	CodeBookNo;
	
	for( CodeBookNo = 0; CodeBookNo < VITERBI_CODEBOOK_NUM; CodeBookNo++ )
	{
		CalculateFeaToCodeBookDist(ViterbiFea,CodeBookNo);
	}
}

void CVITERBI_CODEBOOK::CalculateFeaToCodeBookDist(VITERBI_FEA ViterbiFea,long CodeBookNo)
{
long	FeaDimNo;
CODE_PRECISION	DotProduct;	
	
	for( FeaDimNo = 0; FeaDimNo < VITERBI_FEA_DIM; FeaDimNo++ )
	{
		x[FeaDimNo]=ViterbiFea[FeaDimNo]-pViterbiCodeBook[CodeBookNo].MeanU[FeaDimNo];
	}
/*
	for( FeaDimNo = 0; FeaDimNo < VITERBI_FEA_DIM; FeaDimNo++ )
	{
		DotProduct=0;
		for(i=0;i<VITERBI_FEA_DIM;i++)
		{
			DotProduct+=pViterbiCodeBook[CodeBookNo].InvR[FeaDimNo][i]*x[i];
		}
		y[FeaDimNo]=DotProduct;
	}
	DotProduct = 0;
	for( FeaDimNo = 0; FeaDimNo < VITERBI_FEA_DIM; FeaDimNo++ )
	{
		DotProduct+=x[FeaDimNo]*y[FeaDimNo];
	}
*/
//	用对角阵计算距离
	DotProduct = 0;
	for( FeaDimNo = 0; FeaDimNo < VITERBI_FEA_DIM; FeaDimNo++ )
	{
		DotProduct+=x[FeaDimNo]*x[FeaDimNo]*(pViterbiCodeBook[CodeBookNo].InvR[FeaDimNo][FeaDimNo]);
	}
	
	pFeaToCodeBookProb[CodeBookNo]  = (float)(-0.5*(Log2PI*VITERBI_FEA_DIM + log( fabs(pViterbiCodeBook[CodeBookNo].DetR)) ));	
	pFeaToCodeBookProb[CodeBookNo] -= (float)(0.5*DotProduct);
}

void CVITERBI_CODEBOOK::UpdateNewCode(void)
{
long			i,CodeBookNo;
CODE_PRECISION	Det;

	for( CodeBookNo = 0; CodeBookNo<VITERBI_CODEBOOK_NUM; CodeBookNo++ )
	{
		if(CodeBookCounter[CodeBookNo] < 2 )
			continue;
		for( i=0; i<VITERBI_FEA_DIM;i++ )
		{	//计算特征的均值
			pTmpCodeBook[CodeBookNo].MeanU[i]=pTmpCodeBook[CodeBookNo].MeanU[i]/CodeBookCounter[CodeBookNo];
		}
	}
	for( CodeBookNo = 0; CodeBookNo<VITERBI_CODEBOOK_NUM; CodeBookNo++ )
	{
		if(CodeBookCounter[CodeBookNo] < 2 )
			continue;
//		for( i=0; i<VITERBI_FEA_DIM;i++ )
//		{	//计算特征的协方差阵
//			for(j=0;j<VITERBI_FEA_DIM;j++)
//			{
//				pTmpCodeBook[CodeBookNo].InvR[i][j] /= CodeBookCounter[CodeBookNo];
//				pTmpCodeBook[CodeBookNo].InvR[i][j] -= pTmpCodeBook[CodeBookNo].MeanU[i]*pTmpCodeBook[CodeBookNo].MeanU[j];
//			}
//		}
		//2000，6，5 用对角阵训练
		for( i=0; i<VITERBI_FEA_DIM;i++ )
		{	//计算特征的方差
				pTmpCodeBook[CodeBookNo].InvR[i][i] /= CodeBookCounter[CodeBookNo];
				pTmpCodeBook[CodeBookNo].InvR[i][i] -= pTmpCodeBook[CodeBookNo].MeanU[i]*pTmpCodeBook[CodeBookNo].MeanU[i];
		}
		//	求对角阵逆及行列式值
		Det = 1;
		for( i=0; i<VITERBI_FEA_DIM; i++ )
		{
			Det=Det*pTmpCodeBook[CodeBookNo].InvR[i][i];
			pTmpCodeBook[CodeBookNo].InvR[i][i]=1.0/pTmpCodeBook[CodeBookNo].InvR[i][i];
		}
		pTmpCodeBook[CodeBookNo].DetR=Det;


	}
	SetAllCodeBook(pTmpCodeBook);
}
void CVITERBI_CODEBOOK::ClearTmpCodeBuf(void)
{
long CodeBookNo,i,j;

	for(CodeBookNo=0;CodeBookNo<VITERBI_CODEBOOK_NUM;CodeBookNo++)
	{
		CodeBookCounter[CodeBookNo]=0;
		for(i=0;i<VITERBI_FEA_DIM;i++)
		{
			pTmpCodeBook[CodeBookNo].MeanU[i]=0;
			for(j=0;j<VITERBI_FEA_DIM;j++)
				pTmpCodeBook[CodeBookNo].InvR[i][j]=0;
		}
	}
}
bool CVITERBI_CODEBOOK::LoadCodeBook(char *FileName)
{
FILE	*fp;

	fp=fopen(FileName,"rb");
	if(fp == NULL) return(false);
	fread(pViterbiCodeBook,sizeof(VITERBI_CODEBOOK),VITERBI_CODEBOOK_NUM,fp);
	fclose(fp);
	return(true);
}

bool CVITERBI_CODEBOOK::SaveCodeBook(char *FileName)
{
FILE	*fp;
	fp=fopen(FileName,"wb");
	if(fp == NULL) return(false);
	fwrite(pViterbiCodeBook,sizeof(VITERBI_CODEBOOK),VITERBI_CODEBOOK_NUM,fp);
	fclose(fp);
	return(true);
}
