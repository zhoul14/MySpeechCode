#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include"CWaveFormatFile.h"
#include"GetNewFeature.h"
#include "DatFile.h"
#include <io.h>
#include "../CommonLib/FileFormat/FeatureFileSet.h"
#pragma warning(disable:4996)
using namespace std;

#define DATFILE false


vector<string> readWavList(string filename) {
	vector<string> r;

	ifstream infile(filename);
	string t;
	while (infile >> t) {
		r.push_back(t);
	}
	infile.close();
	return r;
}
void saveBinData(const char* filename, double* buf, int n) {
	FILE* fid = fopen(filename, "wb+");
	if (!fid) {
		printf("cannot open file [%s]\n", filename);
		exit(-1);
	}
	fwrite(buf, sizeof(double), n, fid);
	fclose(fid);
	return;
}
void GetNewFile_1(const char* wavlistfile, string filename){
	//const char* wavlistfile = "file_male.flist";

	string wavlist(wavlistfile);

	vector<string> flist = readWavList(wavlist);

	for (int k = 0; k < flist.size(); k++){

		vector<string>list = readWavList(flist[k]);

		//	="D:\\speech\\863\\male\\d51\\M";
		string new_str = filename;
		if (k < 9)new_str += (char)0 + 48;
		char int_str[3];

		itoa(k + 1, int_str, 10);

		new_str += int_str;

		new_str += +".d48";//d51

		FILE* fid = fopen(new_str.c_str(), "wb+");

		long sum = 0;

		float * * out_features = new float *[list.size()];

		long *out_fNum = new long[list.size()];

		FeatureIndex *out_fIdx = new FeatureIndex[list.size()];

		for (int i = 0; i < list.size(); i++) {

			CWaveFile wav;

			wav.Open(list.at(i).c_str(), CWaveFile::modeRead);

			int byteSize = wav.GetDataByteSize();

			short* samples = new short[byteSize / 2];

			wav.Read(samples, byteSize);

			wav.Close();

			GetNewFeature GNF(samples, byteSize / 2);

			out_features[i] = GNF.GetFeatureBuff();

			out_fNum[i] = GNF.GetFrameNum()*(N_DIM + DIM);

			if (i == 0)
				out_fIdx[i].offset = list.size()*sizeof(FeatureIndex);
			else
				out_fIdx[i].offset = out_fIdx[i - 1].offset + out_fIdx[i - 1].byteSize;

			out_fIdx[i].byteSize = out_fNum[i] * sizeof(float);

			sum += out_fNum[i];
		}
		fwrite(out_fIdx, sizeof(FeatureIndex), list.size(), fid);

		for (int i = 0; i < list.size(); i++){

			fwrite(out_features[i], sizeof(float), out_fNum[i], fid);
			delete[]out_features[i];
		}


		printf("completed No.%2d\n", k);
		cout << new_str << endl;
		delete[]out_fNum;
		delete[]out_fIdx;
		fclose(fid);
	}
	return;

}
string DirMatchMainFile(string filename){
	auto iter = filename.rbegin();
	while (*iter != '#')
		iter++;
	iter++;
	string out;
	while (*iter != '\\'){
		out += *iter;
		iter++;
	}
	reverse(out.begin(), out.end());
	return out;
}
void GetNewFile_2(const char* wavlistfile, string filename, char key, int ong,char* postfix){
	//const char* wavlistfile = "file_male.flist";

	string wavlist(wavlistfile);

	vector<string> flist = readWavList(wavlist);
	int k = 0;
	string preMainName = DirMatchMainFile(*flist.begin());
	string lastMainName = DirMatchMainFile(*flist.rbegin());

	vector<string>list;

	int cnt = 1;

	for (k = 0; k < flist.size(); k++){

		string tempMainName = DirMatchMainFile(flist[k]);

		if (tempMainName == preMainName){
			list.push_back(flist[k]);
			if ((k != (flist.size() - 1)))
				continue;
		}
		printf("WavFile Nums = %d\n", list.size());
		string new_str = filename;

		auto iter = preMainName.begin();
		while (*iter != key)++iter;
		//new_str+=preMainName.substr(iter-preMainName.begin()+1);
		char NameBuf[80];
		if (ong == 2)
			sprintf(NameBuf, "%s%.2d", new_str.c_str(), cnt);
		if (ong == 3)
			sprintf(NameBuf, "%s%.3d", new_str.c_str(), cnt);

		new_str = NameBuf;

		preMainName = tempMainName;

		new_str += +postfix;//d45

		FILE* fid = fopen(new_str.c_str(), "wb+");

		long sum = 0;

		float * * out_features = new float *[list.size()];

		long *out_fNum = new long[list.size()];

		FeatureIndex *out_fIdx = new FeatureIndex[list.size()];
		int i = 0;
		for (i = 0; i < list.size(); i++) {

			short* samples;

			int byteSize;

			if (DATFILE)
			{
				DatFile  dat(list.at(i));
				//byteSize = dat.getSampleNum();
			}
			else
			{
				CWaveFile wav;

				wav.Open(list.at(i).c_str(), CWaveFile::modeRead);

				byteSize = wav.GetDataByteSize();

				samples = new short[byteSize / 2];

				wav.Read(samples, byteSize);

				wav.Close();
			}			
	
			GetNewFeature GNF(samples, byteSize / 2);

			out_features[i] = GNF.GetFeatureBuff();

			if (GNF.d48Ord42())
			{
				out_fNum[i] = GNF.GetFrameNum()*(N_DIM + DIM);
			}
			else
			{
				out_fNum[i] = GNF.GetFrameNum()*(DIM + MIN_DIM);
			}

			if (i == 0)
				out_fIdx[i].offset = list.size()*sizeof(FeatureIndex);
			else
				out_fIdx[i].offset = out_fIdx[i - 1].offset + out_fIdx[i - 1].byteSize;

			out_fIdx[i].byteSize = out_fNum[i] * sizeof(float);

			sum += out_fNum[i];
		}
		fwrite(out_fIdx, sizeof(FeatureIndex), list.size(), fid);

		for (int i = 0; i < list.size(); i++){

			fwrite(out_features[i], sizeof(float), out_fNum[i], fid);
			delete[]out_features[i];
		}
		printf("completed No.%.3d ", cnt);
		cout << new_str << endl;
		delete[]out_fNum;
		delete[]out_fIdx;

		fclose(fid);

		list.clear();
		if (k != flist.size())
			list.push_back(flist[k]);
		cnt++;
	}
	return;

}

void GetNewFile_dat(const char* wavlistfile, string filename, char key){

	string wavlist(wavlistfile);

	vector<string> flist = readWavList(wavlist);
	int k = 0;
	string LastMainName = DirMatchMainFile(*flist.begin());

	vector<string>list;

	for (k = 0; k < flist.size(); k++){

		string tempMainName = DirMatchMainFile(flist[k]);

		if (tempMainName == LastMainName){
			list.push_back(flist[k]);
			continue;
		}
		string new_str = filename;

		auto iter = LastMainName.begin();
		while (*iter != key)++iter;
		new_str += LastMainName.substr(iter - LastMainName.begin() + 1);


		LastMainName = tempMainName;

		new_str += +".d51";

		FILE* fid = fopen(new_str.c_str(), "wb+");

		long sum = 0;

		float * * out_features = new float *[list.size()];

		long *out_fNum = new long[list.size()];

		FeatureIndex *out_fIdx = new FeatureIndex[list.size()];
		int i = 0;
		for (i = 0; i < list.size(); i++) {

			CWaveFile wav;

			wav.Open(list.at(i).c_str(), CWaveFile::modeRead);

			int byteSize = wav.GetDataByteSize();

			short* samples = new short[byteSize / 2];

			wav.Read(samples, byteSize);

			wav.Close();

			GetNewFeature GNF(samples, byteSize / 2);

			out_features[i] = GNF.GetFeatureBuff();

			out_fNum[i] = GNF.GetFrameNum()*(N_DIM + DIM);

			if (i == 0)
				out_fIdx[i].offset = list.size()*sizeof(FeatureIndex);
			else
				out_fIdx[i].offset = out_fIdx[i - 1].offset + out_fIdx[i - 1].byteSize;

			out_fIdx[i].byteSize = out_fNum[i] * sizeof(float);

			sum += GNF.GetFrameNum()*(N_DIM + DIM);
		}
		fwrite(out_fIdx, sizeof(FeatureIndex), list.size(), fid);

		for (int i = 0; i < list.size(); i++){

			fwrite(out_features[i], sizeof(float), out_fNum[i], fid);
			delete[]out_features[i];
		}


		printf("completed No.%3d ", k);
		cout << new_str << endl;
		delete[]out_fNum;
		delete[]out_fIdx;

		fclose(fid);

		list.clear();

	}
	return;

}
void GetWavFile(const char* wavlistfile, string filename){
	string wavlist(wavlistfile);

	vector<string> flist = readWavList(wavlist);
	for (int i = 0; i < flist.size(); i++){
		string newName = filename;
		char NameBuf[50];
		sprintf(NameBuf, "%s%.2d", newName.c_str(), i + 1);
		newName = NameBuf;
		newName += '#';
		DatFile myDatFile(flist[i]);
		for (int j = 0; j < myDatFile.getSentenceNum(); j++){
			string newstr = newName;
			char buffer[80];
			sprintf(buffer, "%s%.4d", newstr.c_str(), j);
			newstr = buffer;
			newstr += ".wav";
			myDatFile.writeWav(newstr, j);
		}
		cout << "completed wav:No." << i << endl;

	}
}

void setSeg(){

	for (int j = 0; j < 2; j++)
	{
		char x = 'M';
		if (j)
		{
			x = 'F';
		}
		for (int i = 1; i <= 50; i++)
		{
			char maskFileName[50];
			sprintf_s(maskFileName, "tag/%c%.2d.tag",x,i);
			FILE* maskFile = fopen(maskFileName, "rb");
			if (maskFile == NULL) {
				//std::cout << (format("mask file [%s] cannot be opened") % maskFileName).str();
				printf("mask file [%s] cannot be opened\n", maskFileName);
				exit(-1);
			}
			int n = _filelength(_fileno(maskFile));
			char * maskFileBuf = new char[n];
			if (n != fread(maskFileBuf, sizeof(char), n, maskFile)) {
				//std::cout << (format("error when read mask file [%s]") % maskFileName).str();
				printf("error when read mask file [%s]\n", maskFileName);
				exit(-1);
			}
			fclose(maskFile);

			sprintf_s(maskFileName, "seg/%c%.2d.seg", x, i);

			maskFile = fopen(maskFileName, "wb");
			fwrite(maskFileBuf, sizeof(long), 1254 * 2, maskFile);

			for (int speechIdx = 0; speechIdx < 1254; speechIdx++)
			{
				MaskIndex* mIdx = (MaskIndex*)(maskFileBuf + speechIdx * sizeof(MaskIndex));
				int* maskData = (int*)(maskFileBuf + mIdx->offset);
				int segNum = mIdx->endpointNum / 2;
				int segData[2];
				for (int k = 0; k < segNum; k++)
				{
					segData[0] = maskData[k * 2]+1;
					segData[1] = maskData[k * 2 + 1];
				}
				fwrite(segData, sizeof(int), 2, maskFile);

			}
			fclose(maskFile);

		}

	}
}

int main(){
	//GetWavFile("datfileMa.list","D:\\speech\\isoword\\male\\wav\\M");
	//GetWavFile("datfileFe.list","D:\\speech\\isoword\\female\\wav\\F");
	//GetNewFile_1("file_female.flist","D:\\speech\\863\\female\\d48\\F");
	//GetNewFile_1("file_male.flist","D:\\speech\\863\\male\\d48\\M");/**/
	










	GetNewFile_2("fileisoword_male.list", "E:\\speech\\isoword\\male\\d42\\M", 'M', 2, ".d42");
	GetNewFile_2("fileisoword_female.list", "E:\\speech\\isoword\\female\\d42\\F", 'F', 2, ".d42");

	/*GetNewFile_2("file_intel_female.list","D:\\speech\\intel\\d48\\bej\\female\\F",'f',3);
	GetNewFile_2("file_intel_male.list","D:\\speech\\intel\\d48\\bej\\male\\M",'m',3);*/
	//GetNewFile_2("file_phrase_female.list","D:\\speech\\SbPhrase\\d45\\f",'f',2);
	//GetNewFile_2("file_phrase_male.list","D:\\speech\\SbPhrase\\d45\\m",'m',2);

}

