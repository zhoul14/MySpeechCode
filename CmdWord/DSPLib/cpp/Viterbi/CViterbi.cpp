#include "..\include\CViterbi.h"

CVITERBI::CVITERBI(CVITERBI_CODEBOOK *pCodeBook,long *pCodeBookNo)
{
long	i;

	pCViterbiCodeBook = pCodeBook;
	pPathProb = new float[VITERBI_STATE_NUM];
	pPath	  = new long[VITERBI_STATE_NUM][VITERBI_STATE_NUM+1];

	if( pCodeBookNo != 0 )
	{
		for(i=0;i<VITERBI_STATE_NUM;i++)
		{
			CodeBookNo[i]=pCodeBookNo[i];
		}
	}
}

CVITERBI::~CVITERBI()
{
	delete	[]pPathProb;
	delete	[]pPath;
}
void CVITERBI::SetCodeBook(CVITERBI_CODEBOOK *pCodeBook)
{
	pCViterbiCodeBook=pCodeBook;
}
void CVITERBI::SetStateCodeBookNo(long *pCodeBookNo)
{
long	i;

	for(i=0;i<VITERBI_STATE_NUM;i++)
	{
		CodeBookNo[i]=pCodeBookNo[i];
	}
}

void	CVITERBI::SetupViterbi(VITERBI_FEA *pViterbiFea)
{
long	ViterbiStateNo;

	FrameNum=1;
	//初始化路径
	pPath[VITERBI_STATE_NUM-1][0]					= 0;
	pPath[VITERBI_STATE_NUM-1][VITERBI_STATE_NUM]   = FrameNum;
	//初始化路径距离
	pPathProb[0]=MIN_VITERBI_PROB;
	for(ViterbiStateNo=1;ViterbiStateNo<VITERBI_STATE_NUM;ViterbiStateNo++)
		pPathProb[ViterbiStateNo]=pPathProb[ViterbiStateNo-1]*1.2f;
	//固定第一帧的端点为第0个状态
	pPathProb[0]=pCViterbiCodeBook->pFeaToCodeBookProb[ CodeBookNo[0] ];
}


float * CVITERBI::FrameSynViterbi(VITERBI_FEA *pViterbiFea)
{
long	i,ViterbiStateNo;

	//开始VITERBI算法,搜索概率（的对数）最大的路径
	for(ViterbiStateNo = VITERBI_STATE_NUM-1; ViterbiStateNo > 0; ViterbiStateNo-- )
	{
		if( pPathProb[ViterbiStateNo-1] > pPathProb[ViterbiStateNo] )
		{
			pPathProb[ViterbiStateNo] = pPathProb[ViterbiStateNo-1];
			//记录进入当前状态的时刻
			pPath[ViterbiStateNo][ViterbiStateNo] = FrameNum;
			//更新当前状态的路径记录
			for(i=ViterbiStateNo-1;i>0;i--)    
				pPath[ViterbiStateNo][i]=pPath[ViterbiStateNo-1][i];
		}
		pPathProb[ViterbiStateNo]+=pCViterbiCodeBook->pFeaToCodeBookProb[CodeBookNo[ViterbiStateNo]];
	}
	//路径在第0个状态驻留
	pPathProb[0]+=pCViterbiCodeBook->pFeaToCodeBookProb[CodeBookNo[0]];
	FrameNum++;
	pPath[VITERBI_STATE_NUM-1][VITERBI_STATE_NUM]  = FrameNum;
	return(pPathProb);
}
long * CVITERBI::GetSegmentation(void)
{
	return(pPath[VITERBI_STATE_NUM-1]);
}

float * CVITERBI::GetPathProb(void)
{
	return(pPathProb);
}
