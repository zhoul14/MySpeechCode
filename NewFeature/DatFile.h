#ifndef WJQ_DAT_FILE_H
#define WJQ_DAT_FILE_H

#include <string>

typedef struct {
	int	offset;
	int byteLen;
} DatFileIdx;

class DatFile {
private:
	int* lens;

	short** data;

	int sentenceNum;

public:
	DatFile(const std::string& filename);

	~DatFile();

	int getSentenceNum() const;

	int getSampleNum(int sidx) const;

	void getSpeech(int sidx, short* buffer) const;

	void writeWav(const std::string& wavName, int sidx) const;
};


#endif