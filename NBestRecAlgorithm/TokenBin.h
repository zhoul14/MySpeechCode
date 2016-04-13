#ifndef WJQ_TOKEN_BIN_H
#define WJQ_TOKEN_BIN_H

#include <list>
#include <vector>
#include <iostream>


#include "../CommonLib/commonvars.h"
#include "../CommonLib/Dict/WordDict.h"
#include "RecToken.h"

class TokenBin {
	friend class NBestRecAlgorithm;

private:
	int cbidx;

	bool isEndingBin;

	std::vector<RecToken*> content;

	std::vector<TokenBin*> previous;

	void addToken(RecToken* t);

	void addPreviousBin(TokenBin* b);

	TokenBin(int cbidx);

	RecToken* getBestToken();

	int tokenNum();


};


#endif