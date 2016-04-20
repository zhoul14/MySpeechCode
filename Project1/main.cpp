#include <iostream>
#include <direct.h> 
#include <stdio.h>
#include "windows.h"
using namespace std;
int fileNum = 50;
double coef = 0.0;
char pCoef[10];
int fdim = 45;
bool bHeadNoise = true;
//PARAM:
//TRAINITER  1
//	DICTCONFIG D:/MyCodes/DDBHMMTasks/Didict/worddict.txt
//	TRAINNUM 50
//	EMITER  13
//	DURWEIGHT  5 
//	TRIPHONE 0
//	FDIM 48
//	INITCODEBOOK D:/myCodeBooks/full_16mix48d_male_log.cb
//	USECUDA 0
//	OUTCODEBOOK  full_1mix48d_prime_log.cb 
void GenerateTrainConfigFile(const char* Filename, int idx , int fDim)
{	
	FILE* fid=fopen(Filename,"w+");
	fprintf(fid,"PARAM:\n");
	fprintf(fid,"TRAINITER  2\n");
	fprintf(fid,"DICTCONFIG D:/MyCodes/DDBHMMTasks/Didict/worddict.txt\n");
	fprintf(fid,"TRAINNUM 49\n");
	fprintf(fid,"EMITER  30\n");
	fprintf(fid,"DURWEIGHT  5\n");
	fprintf(fid,"TRIPHONE 0\n");
	fprintf(fid,"FDIM %d\n",fDim);
	fprintf(fid,"INITCODEBOOK  MLE/full_1mix%dd_male_log%.2d.cb \n",/*(int)(coef*100),*/fDim,idx+1);
	//fprintf(fid,"INITCODEBOOK full_8mix%dd.cb\n",fDim);

	//fprintf(fid,"INITCODEBOOK full_1mix%dd_male.cb\n",fDim);
	fprintf(fid,"USECUDA 1\n");
	fprintf(fid,"COEF %s\n",pCoef);
	fprintf(fid,"OUTCODEBOOK  MMIE/full_1mix%dd_male_log%.2d.cb \n",/*(int)(coef*100),*/fDim,idx+1);
	//fprintf(fid,"OUTCODEBOOK  full_1mix%dd_prime_log%.2d.cb \n",fDim,idx+1);
	fprintf(fid,"DATA:\n");

	for ( int i = 0; i < fileNum; i++ )
	{
		if ( i == idx )
		{
			continue;
		}
		if(i<50)
			fprintf(fid,"E:/Speech/isoword/male/d%d/M%.2d.d%d E:/Speech/isoword/sbword.idx E:/Speech/isoword/male/tag/M%.2d.tag \n",fDim,i+1,fDim,i+1); 
		else
			fprintf(fid,"E:/Speech/isoword/female/d%d/F%.2d.d%d E:/Speech/isoword/sbword.idx E:/Speech/isoword/female/tag/F%.2d.tag \n",fDim,i-50+1,fDim,i+1-50);

	}
	fclose(fid);
}

void GeneratePrimeTrainConfigFile(const char* Filename, int idx , int fDim)
{	

	int iFdim = 45;
	FILE* fid=fopen(Filename,"w+");
	fprintf(fid,"PARAM:\n");
	fprintf(fid,"TRAINITER  3\n");
	fprintf(fid,"DICTCONFIG D:/MyCodes/DDBHMMTasks/Didict/worddict.txt\n");
	fprintf(fid,"TRAINNUM 49\n");
	fprintf(fid,"EMITER  13\n");
	fprintf(fid,"DURWEIGHT  5\n");
	fprintf(fid,"TRIPHONE 0\n");
	fprintf(fid,"FDIM %d\n",iFdim);
	fprintf(fid,"INITCODEBOOK MMIE/full_2mix%dd_female_mmie.cb\n",iFdim);
	//fprintf(fid,"INITCODEBOOK  full_1mix%dd_prime_log%.2d.cb \n",fDim,idx+1);
	fprintf(fid,"USECUDA 1\n");
	fprintf(fid,"OUTCODEBOOK  MMIE/full_2mix%dd_female_log%.2d.cb \n",fDim,idx+1);
	fprintf(fid,"COEF %s\n",pCoef);
	fprintf(fid,"DATA:\n");

	for ( int i = 0; i < fileNum; i++ )
	{
		if ( i == idx )
		{
			continue;
		}
		if(i<50)
			fprintf(fid,"E:/Speech/isoword/female/d%d/F%.2d.d%d E:/Speech/isoword/sbword.idx E:/Speech/isoword/female/tag/F%.2d.tag \n",iFdim,i+1,iFdim,i+1); 
		else
			fprintf(fid,"E:/Speech/isoword/male/d%d/M%.2d.d%d E:/Speech/isoword/sbword.idx E:/Speech/isoword/male/tag/M%.2d.tag\n",iFdim,i-50+1,iFdim,i+1-50);

	}
	fclose(fid);
}


void GenerateRecConfigFile(const char* Filename, int idx , int fDim)
{
	FILE* fid=fopen(Filename,"w+");
	fprintf(fid,"PARAM:\n");
	fprintf(fid,"DURWEIGHT  5 \n");
	fprintf(fid,"DICTCONFIG D:/MyCodes/DDBHMMTasks/Didict/worddict.txt\n");
	fprintf(fid,"RECNUM 1\n");
	fprintf(fid,"TRIPHONE 0\n");
	fprintf(fid,"FDIM %d\n",fDim);//full_1mix48d_prime_log01
	fprintf(fid,"CODEBOOK MMIE/full_1mix%dd_male_log%.2d.cb \n",/*(int)(coef*100),*/fDim,idx+1);
	//fprintf(fid,"CODEBOOK  full_1mix%dd_prime_log%.2d.cb \n",fDim,idx+1);
	fprintf(fid,"USECUDA 1\n");
	fprintf(fid,"SEGMENTMODEL 1\n");
	fprintf(fid,"COEF %s\n",pCoef);
	fprintf(fid,"HEADNOISE %d\n",bHeadNoise);
	fprintf(fid,"MULTIJUMP\nDATA:\n");
	int i = idx;
	if ( i < 50)
		fprintf(fid,"E:/Speech/isoword/male/d%d/M%.2d.d%d E:/Speech/isoword/sbword.idx E:/Speech/isoword/male/tag/M%.2d.tag res/M%.2d.rec\n",fDim,i+1,fDim,i+1,i+1);
	else
		fprintf(fid,"E:/Speech/isoword/female/d%d/F%.2d.d%d E:/Speech/isoword/sbword.idx E:/Speech/isoword/female/tag/F%.2d.tag res/F%.2d.rec\n",fDim,i-50+1,fDim,i-50+1,i-50+1);
	fclose(fid);
}
void GenerateTrainCrossBatFile(const char * batFileName, const char * exeName, const char * TConfigName)
{
	FILE * fid=fopen(batFileName,"w+");
	fprintf(fid,"%s\t%s\t%s\n",exeName,TConfigName,TConfigName);

	fclose(fid);
}
int main()
{	
	unsigned int* x = new unsigned int[2];
	*x = 0xBE7A04D7;
	x[1]= 0xBE7A0506;
	float *fx = (float*)x,*fy = (float *) x + 1;

	printf("ÊäÈëcoef:\n");
	scanf("%lf",&coef);
	printf("\n%f",coef);
	char T_bat[30];
	sprintf(T_bat,"CrossTrain_Coef%.3d.bat",(int)(coef*100));
	char R_bat[30];
	sprintf(R_bat,"CrossRec_Coef%.3d.bat",(int)(coef*100));
	char * TT_Bat="CrossTrain_prime.bat";
	FILE* Tfid=fopen(T_bat,"w+");
	FILE* Rfid=fopen(R_bat,"w+");

	char Tdir[30],Rdir[30];
	sprintf(Tdir,"CrossTrainConfig_Coef%.3d",(int)(coef*100));
	sprintf(Rdir,"CrossRecConfig_Coef%.3d",(int)(coef*100));
	if (GetFileAttributes(Tdir) == INVALID_FILE_ATTRIBUTES) {
		CreateDirectory(Tdir, NULL);
	}
	if (GetFileAttributes(Rdir) == INVALID_FILE_ATTRIBUTES) {
		CreateDirectory(Rdir, NULL);
	}
	if (GetFileAttributes("CrossTrain_prime") == INVALID_FILE_ATTRIBUTES)
	{
		CreateDirectory("CrossTrain_prime",NULL);
	}
	sprintf(pCoef,"%f",coef);
	FILE* TTfid=fopen(TT_Bat,"w+");
	for (int i = 0; i < fileNum; i++)
	{
		char FileNameT[40], FileNameR[40],FileNameTT[40];
		if( i < 50)
		{
			sprintf(FileNameT,"%s/M%.2d.tcon",Tdir,i+1);
			sprintf(FileNameTT,"CrossTrain_prime/M%.2d.tcon",i+1);
			sprintf(FileNameR,"%s/M%.2d.rcon",Rdir,i+1);
		}
		else
		{
			sprintf(FileNameT,"%s/F%.2d.tcon",Tdir,i+1-50);
			sprintf(FileNameTT,"CrossTrain_prime/F%.2d.tcon",i+1-50);
			sprintf(FileNameR,"%s/F%.2d.rcon",Rdir,i+1-50);
		}
		//fprintf(Tfid,"SpeechTrain.exe\t%s\t%s\n",FileNameT,FileNameT);
		fprintf(Tfid,"SpeechTrainCoef.exe\t%s\n",FileNameT);
		fprintf(Rfid,"SimpleSpeechRecCoef.exe\t%s\n",FileNameR);
		fprintf(TTfid,"SpeechTrainPrimeCoef.exe\t%s\t%s\n",FileNameTT,FileNameT);
		GenerateTrainConfigFile(FileNameT,i,fdim);
		GeneratePrimeTrainConfigFile(FileNameTT,i,fdim);
		GenerateRecConfigFile(FileNameR,i,fdim);
	}
	fclose(Tfid);
	fclose(Rfid);
	fclose(TTfid);
	return 0;
}