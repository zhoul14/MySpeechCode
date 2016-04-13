#ifndef _WJQ_SIMPLE_BUFFER_H
#define _WJQ_SIMPLE_BUFFER_H


typedef int (*RecFunc)(short* data, int n);

class SimpleBuffer {
private:
	const static int MAX_BUFFER_LEN = 160000;

	short buf[MAX_BUFFER_LEN];

	int ptr;

	RecFunc rfunc;

public:
	

	const static char SIGNAL_NO_VOICE = 'H';

	const static char SIGNAL_VOICE = 'L';

	SimpleBuffer();

	int readInAndRec(short* data, int n, int flag);

	void setRecFunc(RecFunc func);

	void reset();


};
#endif