#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <io.h>
#include "VAD.h"
#include "../RealTimeRec/RealTimeRec.h"
#include <vector>

long DoRecg(void* pInfo, short* SpeechBuf, long DataNum) {
	static long No = 0;

	printf("检测到语音，开始语音识别．．．\n");
	RealTimeRec* r = (RealTimeRec*)pInfo;
	std::vector<std::vector<SWord> > res0 = r->recSpeech(SpeechBuf, DataNum);
	for (int i = 0; i < res0.size(); i++) {
		for (int j = 0; j < res0[i].size(); j++) {
			if (res0.size() > 1 && j >= 6)
				break;
			printf("%8s ", res0[i][j].label.c_str());
		}
		printf("\n");
	}

	FILE	*fpRec;
	if( DataNum < 0 ) return -2;

	char	strVadFileName[100];
	sprintf_s(strVadFileName,"VAD_%ld.pcm",No++);
	fopen_s(&fpRec,strVadFileName,"w+b");			//将端点检测后的语音存盘
	if(fpRec == NULL ) return -2;
	fwrite(SpeechBuf,sizeof(short),DataNum,fpRec);
	fclose(fpRec);
	printf("%s\n",strVadFileName);


	printf("语音识别结束\n");

	return 0;
}

int main() {
	const char* cbpath = "D:/MyCodes/DDBHMMTasks/codebooks/paper/V00_2mix_w5_full_init.cb";
	const char* dictpath = "D:/MyCodes/SpeechTrainRec/DictConfig/worddict.xml";
	double durWeight = 5;
	double useCuda = true;
	double useSegmentModel = false;
	double useNBest = true;
	RealTimeRec* rec = new RealTimeRec(cbpath, dictpath, durWeight, useCuda, useSegmentModel, useNBest);


	CVAD* vad = new CVAD();
	vad->IniClass(rec);
	vad->SetRecgCallback(DoRecg);


	::fflush(stdin);
	while( _kbhit() == 0 )
	{
		vad->WaitForSpeechDetected();
	}

	delete vad;
	delete rec;
	return 0;
}


