#include "WrongWordAnalysis.h"

#define NBest 20
int main(int argc,char *argv[]){
	
	string outFileName,Configfilename,statfileName;
	if (argc<3)
	{
		outFileName = "WrongRes.txt";
		statfileName = "ErrorRes.txt";
		if(argc<2)
			Configfilename= "stat_config.txt";
		else
			Configfilename=argv[1];
	}
	else if(argc==3)
	{
		outFileName = argv[2];
		Configfilename = argv[1];		
	}	
	else
	{
		printf("error input!");
		exit(-1);
	}
	WrongWordAnalysis my(Configfilename,outFileName,statfileName,"D:/MyCodes/DDBHMMTasks/dict/worddict.txt");
	
	/*ifstream fid(Configfilename.c_str(),ios::in);
	if (!fid)
	{
		printf("can't open config file %s",Configfilename);
		exit(-1);
	}
	string ProcessNumStr;
	getline(fid,ProcessNumStr);
	trimStr(ProcessNumStr);
	int ProcessNum=stoi(ProcessNumStr);
	vector<string>recList(ProcessNum);
	vector<string>ansList(ProcessNum);
	for (int i=0;i<ProcessNum;i++)
	{
		string inString;
		getline(fid,inString);
		trimStr(inString);
		stringstream ss(inString);
		ss>>recList[i]>>ansList[i];
	}
	int outWrongList[1254]={0};
	vector<vector<string>>outWrongIdList(1254);
	FILE * outFid=fopen(outFileName.c_str(),"w+");
	for (int i=0;i<ProcessNum;i++)
	{
		WrongWordAnalysis prf(recList[i],ansList[i],"D:/MyCodes/DDBHMMTasks/dict/worddict.txt");
		fprintf(outFid,"-----------------------------------\n");
		fprintf(outFid,"No.%d rec File:\n",i);
		prf.printWrongRecResToFile(outFid);
		prf.SaveWrongList(outWrongList,outWrongIdList);
	}
	printWrongListToFile(outFid,outWrongList,outWrongIdList);
	fclose(outFid);*/

}

