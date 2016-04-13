#include "WrongWordAnalysis.h"
WrongWordAnalysis::~WrongWordAnalysis()
{
	delete thisdict;
}

void WrongWordAnalysis::clearBuffer()
{
	delete []answerBuf;

	MyresList.clear();
}

void WrongWordAnalysis::printWrongRecResToFile(FILE* fid)
{
	for (int i = 0; i < SentenceNum; i++)
	{
		AnswerIndex* aIdx = (AnswerIndex*)(answerBuf + i * sizeof(AnswerIndex));

		int* start = (int*)(answerBuf + aIdx->offset);

		for (int j = 0; j < aIdx->wordNum; j++) {//Sub Error

			if (MyresList[i][j][0].wordId!=start[j])
			{
				allCnt++;

				PerFileErrorList[FileIdx]++;

				string rw = thisdict->wordToText(MyresList[i][j][0].wordId);

				ErrorMatrix[start[j]][MyresList[i][j][0].wordId]++;

				fprintf(fid,"No.%d WavFile Wrong Rec Result:\n",start[j]);

				WrongList[start[j]]++;

				wrongIdList[start[j]].push_back(rw);

				string aw = thisdict->wordToText(start[j]);

				fprintf(fid,"[%s]\n",aw.c_str());

				for (int k = 0; k < Nbest; k++)
				{		
					rw = ( thisdict->wordToText( MyresList[i][j][k].wordId));

					fprintf( fid, "%s ", rw.c_str());
				}
				fprintf( fid, "\n\n");

				if(thisdict->wordToText(MyresList[i][j][0].wordId)[0] != aw[0])
					ConsonanErrorList[start[j]].push_back(thisdict->wordToText(MyresList[i][j][0].wordId));
				else if ((aw.substr(0,2)=="zh")||(aw.substr(0,2)=="ch")||(aw.substr(0,2)=="sh"))
				{
					if (thisdict->wordToText(MyresList[i][j][0].wordId)[1]!='h')
					{
						ConsonanErrorList[start[j]].push_back(thisdict->wordToText(MyresList[i][j][0].wordId));
					}
					else if((thisdict->wordToText(MyresList[i][j][0].wordId).substr(0,2)=="ch")||(thisdict->wordToText(MyresList[i][j][0].wordId).substr(0,2)=="zh")||(thisdict->wordToText(MyresList[i][j][0].wordId).substr(0,2)=="sh")) 
					{
						if (aw[1]!='h')
							ConsonanErrorList[start[j]].push_back(thisdict->wordToText(MyresList[i][j][0].wordId));
						else
							VowelErrorList[start[j]].push_back(thisdict->wordToText(MyresList[i][j][0].wordId));
					}
				}
				else
					VowelErrorList[start[j]].push_back(thisdict->wordToText(MyresList[i][j][0].wordId));

				int cntPerPy=0;
				for (int k = 0; k < wrongIdList[start[j]].size(); k++)
				{
					if ( wrongIdList[start[j]][k] == rw)
						cntPerPy++;
					if ( cntPerPy > 1)
					{
						wrongIdList[start[j]].erase(wrongIdList[start[j]].end()-1);
					}
				}

			}
		}
		if (MyresList[i].size() > aIdx->wordNum)//Insert Error
		{
			fprintf(fid,"No.%d WavFile Wrong Rec Result:\n",start[0]);

			fprintf(fid,"[%s](Insert Error)\n",thisdict->wordToText(start[0]).c_str());

			string str = thisdict->wordToText(MyresList[i][0][0].wordId)+thisdict->wordToText(MyresList[i][1][0].wordId);

			InsertErrorIdList[start[0]].push_back('['+str+']');

			fprintf( fid, "%s\n", str.c_str());
		}
	}
}

void WrongWordAnalysis::ReadOneFile(string filename,string answerFileName) {

	ifstream fid(filename.c_str());
	if(!fid)
	{
		cout<<"can't open the Rec file"<<filename<<"!!!"<<endl;
		exit(-1);
	}

	string SentenceNumStr;

	getline(fid,SentenceNumStr);

	SentenceNum=stoi(SentenceNumStr);

	for (int j=0;j<SentenceNum;j++)
	{
		string resNumStr;

		getline(fid,resNumStr);

		if (resNumStr.empty())
		{
			getline(fid,resNumStr);
		}

		int resNum=stoi(resNumStr);

		vector<vector<SWord>>*res=new vector<vector<SWord>>;

		for (int i = 0; i < 1; i++) {//1

			string SwordStr;

			getline(fid,SwordStr);

			vector<SWord>*temp = new vector<SWord>;

			stringstream ss(SwordStr);

			for (int k = 0; k < Nbest; k++)
			{
				string IdStr,lhStr;

				ss>>IdStr>>lhStr;
				int ansId=stoi(IdStr);

				SWord *r = new SWord;

				r->wordId=ansId;

				r->lh=stoi(lhStr);

				temp->push_back(*r);
			}

			res->push_back(*temp);
			if (resNum >1)
			{
				getline(fid,SwordStr);
			}

		}

		MyresList.push_back(*res);

	}
	fid.close();

	ifstream answerFileid(answerFileName.c_str(),ios::binary|ios::in);
	if(!answerFileid)
	{
		cout<<"can't open the answer file"<<answerFileid<<"!!!"<<endl;
		exit(-1);
	}
	int n=GetFileSize(answerFileName);
	answerBuf=new char[n];
	answerFileid.read(answerBuf,n);
	answerFileid.close();	
}

int WrongWordAnalysis::GetFileSize(string filename)
{
	ifstream in(filename);

	in.seekg(0,ios::end);

	return in.tellg();
}

WrongWordAnalysis::WrongWordAnalysis(string Configfilename,string outFileName,string statFile,string dictConfigFIle)
{
	ifstream fid(Configfilename.c_str(),ios::in);
	if (!fid)
	{
		printf("can't open config file %s",Configfilename);
		exit(-1);
	}
	thisdict=new WordDict(dictConfigFIle.c_str());

	memset(WrongList,0,1254*sizeof(int));

	wrongIdList.resize(1254);

	allCnt = 0;

	ConsonanErrorList.resize(1254);

	VowelErrorList.resize(1254);

	ErrorMatrix.resize(1254);

	FileIdx = 0;

	for (int i = 0; i < 1254; i++)
	{
		ErrorMatrix[i].resize(1254);
	}

	InsertErrorIdList.resize(1254);

	string ProcessNumStr;

	getline(fid,ProcessNumStr);

	trimStr(ProcessNumStr);

	ProcessNum=stoi(ProcessNumStr);

	vector<string>recList(ProcessNum);

	vector<string>ansList(ProcessNum);

	PerFileErrorList.resize(ProcessNum);

	for (int i=0;i<ProcessNum;i++)
	{
		string inString;
		getline(fid,inString);
		trimStr(inString);
		stringstream ss(inString);
		ss>>recList[i]>>ansList[i];
	}

	FILE * outFid = fopen(outFileName.c_str(),"w+");
	FILE * StatFid = fopen(statFile.c_str(),"w+");
	for (int i=0;i<ProcessNum;i++)
	{
		FileIdx = i;

		ReadOneFile(recList[i],ansList[i]);
		printf("No.%d File Completed\n",i);

		fprintf(outFid,"-----------------------------------\n");

		fprintf(outFid,"No.%d rec File:\n",i);

		printWrongRecResToFile(outFid);

		clearBuffer();
	}

	printWrongListToFile(StatFid);

	fclose(outFid);
	fclose(StatFid);
}

void WrongWordAnalysis::trimStr(string& inStr)
{

	const char* spacelist=" \r\t\n";

	inStr.erase(inStr.find_last_not_of(spacelist)+1);

	inStr.erase(0,inStr.find_first_not_of(spacelist));

}

void WrongWordAnalysis::printWrongListToFile(FILE* fid){

	fprintf(fid,"================== All File Wrong ================\n");

	WordDict dict("D:/MyCodes/DDBHMMTasks/dict/worddict.txt");



	for (int i=0;i<1254;i++)
	{
		if(WrongList[i]){

			fprintf(fid,"[%d]%s:%d\n",i,dict.wordToText(i).c_str(),WrongList[i]);

			for(int j = 0; j < wrongIdList[i].size(); j++)
			{
				int wid = dict.textToWord(wrongIdList[i][j]);
				fprintf(fid,"%s(%d)\t",wrongIdList[i][j].c_str(),ErrorMatrix[i][wid]);

			}

			fprintf(fid,"\n");

			for (int j = 0; j < InsertErrorIdList[i].size(); j++)
			{
				if (j == 0)
					fprintf(fid,"¡ºInsert Error¡»:\n");

				fprintf(fid,"%s\t",InsertErrorIdList[i][j].c_str());

				if (j == InsertErrorIdList[i].size()-1)
					fprintf(fid,"\n");
			}
			fprintf(fid,"\n");
		}
	}
	int cnt = 0;
	fprintf(fid,"Consonant error:\n");
	for (int i = 0; i < 1254; i++)
	{
		int j = 0;
		if (!ConsonanErrorList[i].empty())
		{
			fprintf( fid,"No.%d [%s] :",i,thisdict->wordToText(i).c_str());

			for (j ; j < ConsonanErrorList[i].size(); j++)
			{
				fprintf( fid,"%s\t",ConsonanErrorList[i][j].c_str());
				cnt++;
			}
			fprintf(fid,"\n");
		}		
	}
	int cntV = 0;
	fprintf(fid,"Vowel Error:\n");
	for (int i = 0; i < 1254; i++)
	{
		int j = 0;
		if (!VowelErrorList[i].empty())
		{
			fprintf( fid,"No.%d [%s] :",i,thisdict->wordToText(i).c_str());

			for (j ; j < VowelErrorList[i].size(); j++)
			{
				fprintf( fid,"%s\t",VowelErrorList[i][j].c_str());
				cntV++;
			}
			fprintf(fid,"\n");
		}
	}

	int errorkind=0;
	for (int i = 0; i < 1254; i++)
	{
		for (int j = 0; j < 1254; j++)
		{
			if(ErrorMatrix[i][j])
				errorkind++;
		}
	}

	fprintf(fid,"================== Conclusion ================\n");

	fprintf(fid,"All Consonant error is cnt:%d ,  All Vowel error is cnt:%d,Vall error is :%d , Consonant error rate is :%f\n",cnt,cntV,allCnt,(float)cnt/allCnt);
	fprintf(fid,"ALL ErrorRate:%f\nAll Error type:%d\n",(float)allCnt/ProcessNum/1254,errorkind);
	for (int i = 0; i < ProcessNum; i++)
	{
		fprintf(fid,"No.%2d File ,error %f :\n",i,(float)(1254-PerFileErrorList[i])/1254);
	}
}