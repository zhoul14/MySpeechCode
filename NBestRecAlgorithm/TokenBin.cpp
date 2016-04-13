#include "TokenBin.h"

void TokenBin::addToken(RecToken* t) {
	if (t == NULL) {
		std::cout << "state trace pointer is null";
		exit(-1);
	}
	content.push_back(t);
}

void TokenBin::addPreviousBin(TokenBin* b) {
	if (b == NULL) {
		std::cout << "state trace bin pointer is null";
		exit(-1);
	}
	previous.push_back(b);
}

TokenBin::TokenBin(int cbidx) {
	this->cbidx = cbidx;
}

RecToken* TokenBin::getBestToken() {
	return content.size() == 0 ? NULL : content.front();
}



int TokenBin::tokenNum() {
	return content.size();
}