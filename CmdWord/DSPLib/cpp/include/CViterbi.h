#define	VITERBI_MODEL_NUM	1254	//模型的个数
#define SILENCE_STATE_NUM	2
#define	VITERBI_STATE_NUM	(6+SILENCE_STATE_NUM)	//每个模型的状态数
#define	VITERBI_FEA_DIM		45		//模型特征的维数

#define	VITERBI_CODEBOOK_NUM	(856+SILENCE_STATE_NUM)	//100*2+164*4
#define	MIN_VITERBI_PROB	(-1.0e+4)

typedef float	VITERBI_FEA[VITERBI_FEA_DIM];
typedef double	CODE_PRECISION;
typedef struct tagViterbiCodeBook{
	CODE_PRECISION	MeanU[VITERBI_FEA_DIM];
	CODE_PRECISION	InvR[VITERBI_FEA_DIM][VITERBI_FEA_DIM];
	CODE_PRECISION	DetR;
} VITERBI_CODEBOOK;

class CVITERBI_CODEBOOK
{
	private:
		CODE_PRECISION	x[VITERBI_FEA_DIM],y[VITERBI_FEA_DIM];
		CODE_PRECISION	Log2PI;
	public:
		VITERBI_CODEBOOK *pViterbiCodeBook;
		VITERBI_CODEBOOK *pTmpCodeBook;
		long	CodeBookCounter[VITERBI_CODEBOOK_NUM];

		static long	CoreModelStateToCodeBookNoMap[VITERBI_MODEL_NUM][VITERBI_STATE_NUM-2];
		long	(*ModelStateToCodeBookNoMap)[VITERBI_STATE_NUM];
		float	*pFeaToCodeBookProb;
		CVITERBI_CODEBOOK();
		~CVITERBI_CODEBOOK();
		void SetCodeBook(VITERBI_CODEBOOK *pCodeBook,long CodeBookNo);
		void SetAllCodeBook(VITERBI_CODEBOOK *pCodeBook);
		void CalculateFeaToAllCodeBookDist(VITERBI_FEA ViterbiFea);
		void CalculateFeaToCodeBookDist(VITERBI_FEA ViterbiFea,long CodeBookNo);
		VITERBI_CODEBOOK	*GetCodeBookBuffer(void);
		void UpdateNewCode(void);
		void ClearTmpCodeBuf(void);
		bool SaveCodeBook(char *FileName);
		bool LoadCodeBook(char *FileName);

};

class	CVITERBI
{
	private:		
		float   (*pPathProb);
		long	(*pPath)[VITERBI_STATE_NUM+1];
		long	FrameNum;	//记录当前的帧数
		long	CodeBookNo[VITERBI_STATE_NUM];
	public:
		CVITERBI(CVITERBI_CODEBOOK *pCodeBook=0,long *pCodeBookNo=0);
		~CVITERBI();
		CVITERBI_CODEBOOK	*pCViterbiCodeBook;
		void	SetCodeBook(CVITERBI_CODEBOOK *pCodeBook);
		float	*FrameSynViterbi(VITERBI_FEA *pViterbiFea);
		void	SetupViterbi(VITERBI_FEA *pViterbiFea);
		long	*GetSegmentation(void);
		float	*GetPathProb(void);
		void	SetStateCodeBookNo(long *pCodeBookNo);

};

