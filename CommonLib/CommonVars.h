#ifndef	WJQ_COMMON_VARS_H
#define	WJQ_COMMON_VARS_H

#include <string>

#define TOTAL_MONO_STATE_NUM 857
#define	TOTAL_WORD_NUM 1254
#define TOTAL_SYLLABLE_NUM 408
#define	HMM_STATE_DUR 6

#define INITIAL0 0
#define INITIAL0_C 1
#define INITIAL1 2
#define FINAL0 3
#define FINAL1 4
#define FINAL2 5
#define	FINAL2_C 6
#define FINAL3 7
#define FINAL3_C 8
#define	TAIL_NOISE 9
#define INVALID_CB_TYPE 10

#define DI_INITIAL0 0
#define DI_INITIAL1 1
#define DI_FINAL0 2
#define DI_FINAL1 3
#define DI_FINAL2 4
#define DI_FINAL3 5
#define DI_TAIL_NOISE 6
#define DI_INVALID_CB_TYPE 7
 
#define STATE_NUM_IN_DI 6
#define	STATE_NUM_IN_WORD 9

#define	INITIAL_WORD_NUM 100
#define	FINAL_WORD_NUM 164

#define	INITIAL_WORD_STATE_NUM 2
#define	FINAL_WORD_STATE_NUM 4

#define	C_CLASS_NUM 27
#define	V_CLASS_NUM 29

#define NO_PREVIOUS_WORD -1
#define NO_NEXT_WORD -1

#define BEST_N 20
#define FEATURE_DIM 45


struct SWord {
	int wordId;
	double lh;
	int endTime;
	int jumpTime[7];
	std::string label;
};


#endif