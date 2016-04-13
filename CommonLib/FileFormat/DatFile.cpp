#include "DatFile.h"
#include "CWaveFormatFile.h"

DatFile::DatFile(const std::string& filename) {
	const int FileHeadSkip = 200;
	const int MaxSentenceNum = 2000;
	
	FILE* fp = fopen(filename.c_str(),"rb");
	if (!fp) {
		printf("cannot open the file %s\n",filename.c_str());
		exit(-1);
	}
	
	DatFileIdx headers[MaxSentenceNum];
	fseek(fp, FileHeadSkip, 0);	
	fread(&headers[0], sizeof(DatFileIdx), 1, fp);

	sentenceNum = (headers[0].offset - FileHeadSkip) / sizeof(DatFileIdx);
	if (sentenceNum > MaxSentenceNum) {
		printf("Error: Too many sentence in the speech file: %ld\n", sentenceNum);
		exit(-1);
	}
	fread(&headers[1], sizeof(DatFileIdx), sentenceNum - 1, fp);

	lens = new int[sentenceNum];
	data = new short*[sentenceNum];
	for (int i = 0; i < sentenceNum; i++) {
		lens[i] = headers[i].byteLen / sizeof(short);
		data[i] = new short[lens[i]];

		fseek(fp, headers[i].offset, SEEK_SET);
		fread(data[i], sizeof(short), lens[i], fp);
	}
	fclose(fp);

}

DatFile::~DatFile() {
	for (int i = 0; i < sentenceNum; i++) {
		delete [] data[i];
	}
	delete [] data;
	delete [] lens;
}

int DatFile::getSentenceNum() const {
	return sentenceNum;
}

int DatFile::getSampleNum(int sidx) const {
	if (sidx < 0 || sidx >= sentenceNum) {
		printf("sidx[%d] out of range in DatFile::getSampleNum, sentenceNum = %d\n", sidx, sentenceNum);
		exit(-1);
	}


	int r = lens[sidx];
	return r;
}

void DatFile::getSpeech(int sidx, short* buffer) const {
	if (sidx < 0 || sidx >= sentenceNum) {
		printf("sidx[%d] out of range in DatFile::getSpeech, sentenceNum = %d\n", sidx, sentenceNum);
		exit(-1);
	}
	memcpy(buffer, data[sidx], lens[sidx] * sizeof(short));
}

void DatFile::writeWav(const std::string& wavName, int sidx) const {
	if (sidx < 0 || sidx >= sentenceNum) {
		printf("sidx[%d] out of range in DatFile::writeWav, sentenceNum = %d\n", sidx, sentenceNum);
		exit(-1);
	}

	CWaveFile wav;
	if (!wav.Open(wavName.c_str(), CWaveFile::modeReadWrite)) {
		printf("cannot open wav file [%s]\n", wavName.c_str());
		exit(-1);
	}
	int channelNum = 1;
	int sampleRate = 16000;
	int bitsPerSample = sizeof(short) * 8;
	wav.SetWaveFormat(channelNum, sampleRate, bitsPerSample);
	wav.Write(data[sidx], lens[sidx] * sizeof(short));
	wav.Close();

}


