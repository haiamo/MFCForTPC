/**********
2019/4/4 TonyHe Added PlaneDivision base class.
**********/
#pragma once
#include <algorithm>
#include <vector>

#define ACCURACY 10e-6

using namespace std;

class PlaneDivision
{
public:
	PlaneDivision();
	virtual ~PlaneDivision();

protected:
	vector<double> m_xIntervals;
	vector<double> m_yIntervals;
	double m_EqualThreshold;

public:
	void SetCompareThreshold(double inThrs);

	void GetIntervalNums(size_t& xInts, size_t& yInts);

	vector<double>& GetIntervalList(const char axisName);

public://Member virtual functions:
	virtual void GenerateRandomInterval(unsigned int partNum, double axisBeg, double axisEnd, char axisName);

	virtual void GenerateAllRandIntervals(unsigned int partX, double axisXBeg, double axisXEnd, unsigned int partY, double axisYBeg, double axisYEnd);

	virtual void MoveAllIntervals(vector<double> xAxisM, vector<double> yAxisM);

	virtual void GetPieceID(double* pointPtr, size_t &planeID);
public://Operators:
	bool operator==(const PlaneDivision &rhs);

	bool operator!=(const PlaneDivision &rhs);

	PlaneDivision &operator=(const PlaneDivision &rhs);
};

class PDYMajor :
	public PlaneDivision
{
private:
	vector<vector<double>> m_PieceXIntervals;

public:
	PDYMajor();
	~PDYMajor();

	void GenerateRandomInterval(unsigned int partNum, double axisBeg, double axisEnd, char axisName) override;

	void GenerateAllRandIntervals(unsigned int partX, double axisXBeg, double axisXEnd, unsigned int partY, double axisYBeg, double axisYEnd) override;

	void MoveAllIntervals(vector<double> xAxisM, vector<double> yAxisM) override;

	void GetPieceID(double* pointPtr, size_t &planeID) override;

	PDYMajor &operator=(const PDYMajor &rhs);
};
