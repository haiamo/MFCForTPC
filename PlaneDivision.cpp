#include "stdafx.h"
#include "PlaneDivision.h"


PlaneDivision::PlaneDivision()
{
	m_EqualThreshold = ACCURACY;
}


PlaneDivision::~PlaneDivision()
{
}

void PlaneDivision::SetCompareThreshold(double inThrs)
{
	m_EqualThreshold = inThrs;
}

void PlaneDivision::GetIntervalNums(size_t & xInts, size_t & yInts)
{
	xInts = m_xIntervals.size();
	yInts = m_yIntervals.size();
}

vector<double>& PlaneDivision::GetIntervalList(const char axisName)
{
	switch (axisName)
	{
	case 'x':
	case 'X':
		return this->m_xIntervals;
		break;
	case 'y':
	case 'Y':
		return this->m_yIntervals;
		break;
	default:
		return vector<double>();
		break;
	}
}

void PlaneDivision::GenerateRandomInterval(unsigned int partNum, double axisBeg, double axisEnd, char axisName)
{
	double tmpVal = 0.0;
	srand(time(0));
	vector<double> tmpVec;
	tmpVec.reserve(partNum + 1);

	tmpVec.push_back(axisBeg);
	tmpVec.push_back(axisEnd);
	for (unsigned int ii = 0; ii < partNum - 1; ii++)
	{
		tmpVal = (double)(rand() / double(RAND_MAX)*(axisEnd - axisBeg) + axisBeg);
		tmpVec.push_back(tmpVal);
	}
	sort(tmpVec.begin(), tmpVec.end());

	switch (axisName)
	{
	case 'x':
	case 'X':
		/*if (m_xIntervals.size() > 0)
		{
			m_xIntervals.clear();
		}
		m_xIntervals.reserve(partNum + 1);

		m_xIntervals.push_back(axisBeg);
		m_xIntervals.push_back(axisEnd);
		for (unsigned int ii = 0; ii < partNum - 1; ii++)
		{
			tmpVal = (double)(rand() / double(RAND_MAX)*(axisEnd - axisBeg) + axisBeg);
			m_xIntervals.push_back(tmpVal);
		}
		sort(m_xIntervals.begin(), m_xIntervals.end());*/
		m_xIntervals.swap(tmpVec);
		break;
	case 'y':
	case 'Y':
		/*if (m_yIntervals.size() > 0)
		{
			m_yIntervals.clear();
		}
		m_yIntervals.reserve(partNum + 1);

		m_yIntervals.push_back(axisBeg);
		m_yIntervals.push_back(axisEnd);
		for (unsigned int ii = 0; ii < partNum - 1; ii++)
		{
			tmpVal = (double)(rand() / double(RAND_MAX)*(axisEnd - axisBeg) + axisBeg);
			m_yIntervals.push_back(tmpVal);
		}
		sort(m_yIntervals.begin(), m_yIntervals.end());*/
		m_yIntervals.swap(tmpVec);
		break;
	}
}

void PlaneDivision::GenerateAllRandIntervals(unsigned int partX, double axisXBeg, double axisXEnd, unsigned int partY, double axisYBeg, double axisYEnd)
{
	this->GenerateRandomInterval(partX, axisXBeg, axisXEnd, 'x');
	this->GenerateRandomInterval(partY, axisYBeg, axisXEnd, 'y');
}

void PlaneDivision::MoveAllIntervals(vector<double> xAxisM, vector<double> yAxisM)
{

}

void PlaneDivision::GetPieceID(double * pointPtr, size_t & planeID)
{
}

bool PlaneDivision::operator==(const PlaneDivision & rhs)
{
	bool bEqu = true;
	if (this->m_xIntervals.size() != rhs.m_xIntervals.size() || this->m_yIntervals.size() != rhs.m_yIntervals.size())
	{
		bEqu = false;
	}
	else
	{
		for (size_t ii = 1; ii < this->m_yIntervals.size() - 1; ii++)
		{
			if (abs(this->m_yIntervals[ii] - rhs.m_yIntervals[ii]) < this->m_EqualThreshold)
			{
				bEqu = false;
				break;
			}
		}

		if (bEqu)
		{
			for (size_t ii = 1; ii < this->m_xIntervals.size() - 1; ii++)
			{
				if (abs(this->m_xIntervals[ii] - rhs.m_xIntervals[ii]) < this->m_EqualThreshold)
				{
					bEqu = false;
					break;
				}
			}
		}
	}
	return bEqu;
}

bool PlaneDivision::operator!=(const PlaneDivision & rhs)
{
	return !(*this == rhs);
}

PlaneDivision & PlaneDivision::operator=(const PlaneDivision & rhs)
{
	this->m_EqualThreshold = rhs.m_EqualThreshold;
	this->m_xIntervals = rhs.m_xIntervals;
	this->m_yIntervals = rhs.m_yIntervals;
	return *this;
}

PDYMajor::PDYMajor()
{
}


PDYMajor::~PDYMajor()
{
}

void PDYMajor::GenerateRandomInterval(unsigned int partNum, double axisBeg, double axisEnd, char axisName)
{
	double tmpVal = 0.0;
	size_t folders = 0;
	vector<double> tmpVct, singlVct(1,0.0);
	switch (axisName)
	{
	case 'x':
	case 'X':
		if (m_xIntervals.size() > 0)
		{
			m_xIntervals.clear();
		}
		if (m_yIntervals.size() <= 2)
		{
			folders = 1;
		}
		else
		{
			folders = m_yIntervals.size() - 1;
		}
		m_xIntervals.reserve((partNum - 1)*folders + 2);

		tmpVct.reserve(partNum - 1);
		for (size_t ii = 0; ii < partNum - 1; ii++)
		{
			tmpVal = double(rand()) / double(RAND_MAX)*(axisEnd - axisBeg) + axisBeg;
			tmpVct.push_back(tmpVal);
		}
		::sort(tmpVct.begin(), tmpVct.end());
		m_xIntervals.push_back(axisBeg);
		singlVct[0] = axisBeg;
		m_PieceXIntervals.push_back(singlVct);
		for (size_t ii = 0; ii < folders; ii++)
		{
			m_xIntervals.insert(m_xIntervals.end(), tmpVct.begin(), tmpVct.end());
			m_PieceXIntervals.push_back(tmpVct);
		}
		m_xIntervals.push_back(axisEnd);
		singlVct[0] = axisEnd;
		m_PieceXIntervals.push_back(singlVct);
		break;
	case 'y':
	case 'Y':
		PlaneDivision::GenerateRandomInterval(partNum, axisBeg, axisEnd, 'y');
		break;
	}
}

void PDYMajor::GenerateAllRandIntervals(unsigned int partX, double axisXBeg, double axisXEnd, unsigned int partY, double axisYBeg, double axisYEnd)
{
	this->GenerateRandomInterval(partY, axisYBeg, axisYEnd, 'y');
	this->GenerateRandomInterval(partX, axisXBeg, axisXEnd, 'x');
}

void PDYMajor::MoveAllIntervals(vector<double> xAxisM, vector<double> yAxisM)
{
	double tmpVal = 0.0;
	if (xAxisM.size() > 0)
	{
		vector<vector<double>>::iterator pieceIt = m_PieceXIntervals.begin() + 1;
		vector<double>::iterator moveIt = xAxisM.begin();
		vector<double> tmpVec;
		double xBeg = m_xIntervals[0], xEnd = *prev(m_xIntervals.end());
		while (pieceIt < m_PieceXIntervals.end() - 1)
		{
			if (moveIt >= xAxisM.end())
			{
				break;
			}
			tmpVec.swap(vector<double>());
			tmpVec = *pieceIt;
			vector<double>::iterator curIt = tmpVec.begin();
			while (curIt < tmpVec.end())
			{
				if (moveIt < xAxisM.end())
				{
					*curIt += *moveIt;
				}
				else
				{
					break;
				}
				moveIt++;
				curIt++;
			}
			::sort(tmpVec.begin(), tmpVec.end());
			for (size_t ii = 0; ii < tmpVec.size(); ii++)
			{
				tmpVal = tmpVec[ii];
				if (tmpVal - xBeg < m_EqualThreshold || xEnd - tmpVal < m_EqualThreshold)
				{
					tmpVec.erase(tmpVec.begin() + ii);
					ii--;
				}
				else if (ii > 0)
				{
					if (tmpVal - tmpVec[ii-1] < m_EqualThreshold)
					{
						tmpVec[ii - 1] = (tmpVec[ii - 1] + tmpVal) / 2;
						tmpVec.erase(tmpVec.begin() + ii);
						ii--;
					}
				}
			}
			(*pieceIt).swap(tmpVec);
			pieceIt++;
		}
	}

	if (yAxisM.size() > 0)
	{
		vector<double>::iterator moveIt = yAxisM.begin();
		vector<double>::iterator yIt = m_yIntervals.begin() + 1;
		while (yIt < m_yIntervals.end() - 1)
		{
			if (moveIt < yAxisM.end())
			{
				*yIt += *moveIt;
			}
			else
			{
				break;
			}
			yIt++;
			moveIt++;
		}
		::sort(m_yIntervals.begin(), m_yIntervals.end());
		
		for (size_t ii = 1; ii < m_yIntervals.size(); ii++)
		{
			tmpVal = m_yIntervals[ii];
			if (ii < m_yIntervals.size() - 1)
			{
				if (tmpVal - m_yIntervals[0] < m_EqualThreshold)
				{
					m_yIntervals.erase(m_yIntervals.begin() + ii);
					m_PieceXIntervals.erase(m_PieceXIntervals.begin() + ii);
					ii--;
				}
				else if (tmpVal - m_yIntervals[ii-1] < m_EqualThreshold)
				{
					m_yIntervals[ii - 1] = (tmpVal + m_yIntervals[ii - 1]) / 2;
					m_yIntervals.erase(m_yIntervals.begin() + ii);
					m_PieceXIntervals.erase(m_PieceXIntervals.begin() + ii);
					ii--;
				}
			}
			else
			{
				if (tmpVal - m_yIntervals[ii - 1] < m_EqualThreshold)
				{
					m_yIntervals.erase(m_yIntervals.end() - 2);
					m_PieceXIntervals.erase(m_PieceXIntervals.end() - 2);
					break;
				}
			}
		}
	}

	vector<double> newXInt;
	for (vector<vector<double>>::iterator pieceIt = m_PieceXIntervals.begin(); pieceIt < m_PieceXIntervals.end(); pieceIt++)
	{
		if (newXInt.size() > 0)
		{
			newXInt.insert(newXInt.end(), (*pieceIt).begin(), (*pieceIt).end());
		}
		else
		{
			newXInt = *pieceIt;
		}
	}
	m_xIntervals.swap(newXInt);
}

void PDYMajor::GetPieceID(double * pointPtr, size_t & planeID)
{
	double x = pointPtr[0], y = pointPtr[1];
	size_t xId = 0, yId = 0, totalID = 0;
	if (y - m_yIntervals[0] <= ACCURACY)
	{
		yId = 1;
	}
	else
	{
		for (size_t ii = 1; ii < m_yIntervals.size(); ii++)
		{
			if (y > m_yIntervals[ii - 1] && y <= m_yIntervals[ii])
			{
				yId = ii;
				break;
			}
		}
	}

	vector<double> curXInt = m_PieceXIntervals[yId];
	double xBeg = m_xIntervals[0], xEnd = *prev(m_xIntervals.end());
	if (x - xBeg < ACCURACY)
	{
		xId = 0;
	}
	else
	{
		for (size_t ii = 0; ii <= curXInt.size(); ii++)
		{
			if (0 == ii)
			{
				if (x > xBeg && x <= curXInt[ii])
				{
					xId = 0;
					break;
				}
			}
			else if (curXInt.size() == ii)
			{
				if (x > curXInt[ii - 1] && x <= xEnd)
				{
					xId = curXInt.size();
					break;
				}
			}
			else if (x > curXInt[ii - 1] && x <= curXInt[ii])
			{
				xId = ii;
				break;
			}
		}
	}

	for (size_t ii = 1; ii < yId; ii++)
	{
		totalID += (m_PieceXIntervals[ii].size() + 1);
	}
	totalID += xId;
	planeID = totalID;
}

PDYMajor & PDYMajor::operator=(const PDYMajor & rhs)
{
	this->m_EqualThreshold = rhs.m_EqualThreshold;
	this->m_PieceXIntervals = rhs.m_PieceXIntervals;
	this->m_xIntervals = rhs.m_xIntervals;
	this->m_yIntervals = rhs.m_yIntervals;
	return *this;
}
