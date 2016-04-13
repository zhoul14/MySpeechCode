#include "SimpleBuffer.h"
#include <string.h>

SimpleBuffer::SimpleBuffer() {
	memset(buf, 0, MAX_BUFFER_LEN * sizeof(short));
	ptr = 0;
	rfunc = NULL;
}

void SimpleBuffer::setRecFunc(RecFunc func) {
	this->rfunc = func;
}

int SimpleBuffer::readInAndRec(short* data, int n, int flag) {
	if (n >= MAX_BUFFER_LEN || n <= 0 || data == NULL)
		return -2;

	if (flag == SIGNAL_NO_VOICE) {
		if (ptr > 0) {
			int r = rfunc(buf, ptr);
			ptr = 0;
			return r;
		} else 
			return -1;
	} else if (flag == SIGNAL_VOICE) {
		if (ptr + n >= MAX_BUFFER_LEN) {
			int r = rfunc(buf, ptr);
			memcpy(buf, data, n * sizeof(short));
			ptr = n;
			return r;
		} else {
			memcpy(buf + ptr, data, n * sizeof(short));
			ptr += n;
			return -1;
		}
		
	}
	return -100;
}

void SimpleBuffer::reset() {
	memset(buf, 0, MAX_BUFFER_LEN * sizeof(short));
	ptr = 0;
}