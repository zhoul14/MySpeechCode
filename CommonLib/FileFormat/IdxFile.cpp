#include "IdxFile.h"
#include <io.h>
#include "FeatureFileSet.h"

std::vector< std::vector<int> > IdxFile::fromIdxFile(const std::string& filename) {
	return fromIdxFile(filename.c_str());
}

std::vector< std::vector<int> > IdxFile::fromIdxFile(const char* filename) {
	FILE* answerFile = fopen(filename, "rb");
	if (answerFile == NULL) {
		printf("idx file [%s] cannot be opened\n", filename);
		exit(-1);
	}
	int n = _filelength(_fileno(answerFile));
	char* answerFileBuf = new char[n];
	if (n != fread(answerFileBuf, sizeof(char), n, answerFile)) {
		printf("error when read idx file [%s]\n", filename);
		exit(-1);
	}
	fclose(answerFile);

	AnswerIndex* ai = (AnswerIndex*)answerFileBuf;
	int speechNum = ai->offset / sizeof(AnswerIndex);

	AnswerIndex* p = ai;
	std::vector< std::vector<int> > r(speechNum);
	for (int i = 0; i < speechNum; i++) {
		int* ansList = (int*)(answerFileBuf + p->offset);
		for (int j = 0; j < p->wordNum; j++) {
			r.at(i).push_back(ansList[j]);
		}
		p++;
	}
	return r;
}