#ifndef WJQ_REC_FILE_H
#define WJQ_REC_FILE_H

#include <stdio.h>
#include <string>
#include "../Dict/WordDict.h"

class RecFile {
private:
	int sentenceNum;

	int* sentenceWordNum;

	int** wordId;

	double** wordLh;

	std::string filename;

	int bestN;

public:

	static const int NOCAND = -2;

	RecFile(const std::string& filename, int bestN);

	int getSentenceNum() const;

	int getBestN() const;

	int getSentenceWordNum(int sidx) const;

	void getRecResult(int sidx, int* wordId, double* wordLh) const;

	void getBestKRecResult(int sidx, int K, int* wordId, double* wordLh) const;

	void toRecFile(const std::string& filename) const;

	void toTextFile(const std::string& filename, const WordDict* dict) const;

	~RecFile();

};

#endif