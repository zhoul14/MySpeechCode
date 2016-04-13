#ifndef WJQ_IDX_FILE_H
#define WJQ_IDX_FILE_H

#include <vector>
#include <string>

class IdxFile {
public:
	static std::vector< std::vector<int> > fromIdxFile(const char* filename);

	static std::vector< std::vector<int> > fromIdxFile(const std::string& filename);
};

#endif