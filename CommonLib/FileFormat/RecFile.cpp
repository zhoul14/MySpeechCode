#include "RecFile.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>



RecFile::RecFile(const std::string& filename, int bestN) {
	using namespace std;

	this->filename = filename;
	this->bestN = bestN;

	ifstream input(filename);
	if (!input) {
		printf("cannot open rec file [%s]\n", filename.c_str());
		exit(-1);
	}

	input >> this->sentenceNum;
	sentenceWordNum = new int[sentenceNum];
	memset(sentenceWordNum, 0, sentenceNum * sizeof(int));

	wordId = new int*[sentenceNum];
	memset(wordId, 0, sentenceNum * sizeof(int));

	wordLh = new double*[sentenceNum];
	memset(wordLh, 0, sentenceNum * sizeof(int));

	int sLen = 0;
	string line;
	for (int i = 0; i < sentenceNum; i++) {
		
		input >> sLen;
		input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

		sentenceWordNum[i] = sLen;
		wordId[i] = new int[sLen * bestN];
		wordLh[i] = new double[sLen * bestN];

		for (int j = 0; j < sLen; j++) {
			if (!getline(input, line)) {
				printf("get line error in sentence[%d], word[%d]\n", i, j);
			}
			
// 			if (line == "")
// 				continue;

			istringstream ss(line);

			int k;
			for (k = 0; k < bestN; k++) {
				if (!(ss >> wordId[i][j * bestN + k])) {
					break;
				}
				if (!(ss >> wordLh[i][j * bestN + k])){
					printf("rec file error[wlh] in sentence[%d], word[%d], choice[%d]\n", i, j, k);
					exit(-1);
				}
			}

			while (k < bestN) {
				wordId[i][j * bestN + k] = NOCAND;
				wordLh[i][j * bestN + k] = 0;
				k++;
			}

		}
	}
	input.close();


}

int RecFile::getSentenceNum() const {
	return sentenceNum;
}

int RecFile::getBestN() const {
	return bestN;
}

int RecFile::getSentenceWordNum(int sidx) const {
	return sentenceWordNum[sidx];
}

void RecFile::getRecResult(int sidx, int* wordId, double* wordLh) const{
	int wnum = sentenceWordNum[sidx];
	int cpnum = wnum * bestN;
	memcpy(wordId, this->wordId[sidx], cpnum * sizeof(int));
	memcpy(wordLh, this->wordLh[sidx], cpnum * sizeof(double));
}

RecFile::~RecFile() {
	
	delete [] sentenceWordNum;
	for (int i = 0; i < sentenceNum; i++) {
		delete [] wordId[i];
		delete [] wordLh[i];
	}

	delete [] wordId;
	delete [] wordLh;
}

void RecFile::getBestKRecResult(int sidx, int K, int* wordId, double* wordLh) const {

	if (K > bestN) {
		K = bestN;
	}

	int wnum = sentenceWordNum[sidx];
	for (int i = 0; i < wnum; i++) {
		int* allWordId = this->wordId[sidx];
		double* allWordLh = this->wordLh[sidx];
		for (int j = 0; j < K; j++) {
			wordId[i * K + j] = allWordId[i * bestN + j];
			wordLh[i * K + j] = allWordLh[i * bestN + j];
		}
	}
}

void RecFile::toRecFile(const std::string& filename) const {
	FILE* fid = fopen(filename.c_str(), "w");
	if (!fid) {
		printf("cannot open file [%s] in RecFile::toRecFile", filename.c_str());
		exit(-1);
	}
	fprintf(fid, "%4d\n", this->sentenceNum);
	for (int i = 0; i < sentenceNum; i++) {
		fprintf(fid, "%4d\n", sentenceWordNum[i]);
		for (int j = 0; j < sentenceWordNum[i]; j++) {
			for (int k = 0; k < bestN; k++) {
				fprintf(fid, "%4d %.2f ", wordId[i][j * bestN + k], wordLh[i][j * bestN + k]);
			}
			fprintf(fid, "\n");
			
		}
		fprintf(fid, "\n");
	}

	fclose(fid);
}

void RecFile::toTextFile(const std::string& filename, const WordDict* dict) const {
	FILE* fid = fopen(filename.c_str(), "w");
	if (!fid) {
		printf("cannot open file [%s] in RecFile::toTextFile", filename.c_str());
		exit(-1);
	}
	fprintf(fid, "%4d\n", this->sentenceNum);
	for (int i = 0; i < sentenceNum; i++) {
		fprintf(fid, "IDX_%d WORDNUM_%d\n", i, sentenceWordNum[i]);
		for (int j = 0; j < sentenceWordNum[i]; j++) {
			for (int k = 0; k < bestN; k++) {
				int wid = wordId[i][j * bestN + k];
				double wlh = wordLh[i][j * bestN + k];
				if (wid >= 0) {
					std::string t = dict->wordToText(wid);
					fprintf(fid, "%s\t[%.2f]\t", t.c_str(), wlh);
				}
				
			}
			fprintf(fid, "\n");

		}
		fprintf(fid, "\n");
	}

	fclose(fid);



}