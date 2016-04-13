#ifndef WJQ_LATTICE_NODE_H
#define WJQ_LATTICE_NODE_H

#include <vector>

struct NID {
	int x;

	int y;

	NID(int xx, int yy): x(xx), y(yy){
	}
	NID() {
	}
};

struct LatticeNode {
	std::vector<NID> prev;

	std::vector<double> prevLhDiff;

	int wordId;

	double endLh;

	NID id;

	int prevNum() const {
		return prev.size();
	}

	bool hasPrevious() const {
		bool b = prev.size() > 0 && prev[0].x >= 0;
		return b;
	}

	int endTime() {
		return id.x;
	}
};
typedef std::vector<std::vector<LatticeNode> > Lattice;


#endif