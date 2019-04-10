#include "stdafx.h"
#include "PlaneDivision.h"


PlaneDivision::PlaneDivision()
{
	m_EqualThreshold = 0.0;
}


PlaneDivision::~PlaneDivision()
{
}

void PlaneDivision::SetCompareThreshold(double inThrs)
{
	m_EqualThreshold = inThrs;
}

void PlaneDivision::GenerateRandomInterval(unsigned int partNum, double axisBeg, double axisEnd, char axisName)
{
	double tmpVal = 0.0;
	srand(time(0));
	switch (axisName)
	{
	case 'x':
	case 'X':
		if (m_xIntervals.size() > 0)
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
		sort(m_xIntervals.begin(), m_xIntervals.end());
		break;
	case 'y':
	case 'Y':
		if (m_yIntervals.size() > 0)
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
		sort(m_yIntervals.begin(), m_yIntervals.end());
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
	double tmpVal = 0.0, swithVal = 0.0;
	if (xAxisM.size() > 0)
	{
		vector<double> resInt;
		resInt.reserve(m_xIntervals.size());
		vector<double>::iterator xIntIt = m_xIntervals.begin() + 1;
		vector<double>::iterator xMoveIt = xAxisM.begin();
		while (xIntIt < prev(m_xIntervals.end()))
		{
			tmpVal = *xIntIt + *xMoveIt;
			if (tmpVal - m_xIntervals[0]>m_EqualThreshold && tmpVal - *prev(m_xIntervals.end())<m_EqualThreshold)
			{
				if (tmpVal - *prev(resInt.end()) > m_EqualThreshold)
				{
					resInt.push_back(tmpVal);
				}
				else if (abs(tmpVal - *prev(resInt.end())) < m_EqualThreshold)
				{
					*prev(resInt.end()) = (tmpVal + *prev(resInt.end())) / 2;
				}
				else if (tmpVal - *prev(resInt.end()) < m_EqualThreshold)
				{
					swithVal = *prev(resInt.end());
					*prev(resInt.end()) = tmpVal;
					resInt.push_back(swithVal);
				}
			}
			xIntIt++;
			if (xMoveIt >= xAxisM.end())
			{
				break;
			}
			else
			{
				xMoveIt++;
			}
		}
		resInt.push_back(*prev(m_xIntervals.end()));
		m_xIntervals.swap(resInt);
	}

	if (yAxisM.size() > 0)
	{
		for (size_t ii = 1; ii < min(yAxisM.size(), m_yIntervals.size()); ii++)
		{
			tmpVal = m_yIntervals[ii] + yAxisM[ii];
			if (tmpVal<m_yIntervals[0] || tmpVal>*prev(m_yIntervals.end()))
			{
				m_yIntervals.erase(m_yIntervals.begin() + ii);
				ii--;
			}
			if (ii >= 2)
			{
				if (m_yIntervals[ii] - m_yIntervals[ii - 1] < m_EqualThreshold)
				{
					m_yIntervals[ii - 1] = (m_yIntervals[ii] + m_yIntervals[ii - 1]) / 2;
					m_yIntervals.erase(m_yIntervals.begin() + ii);
					ii--;
				}
			}
		}
		sort(m_yIntervals.begin(), m_yIntervals.end());
	}
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
	vector<double> tmpVct;
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

		m_xIntervals.push_back(axisBeg);
		for (size_t ii = 0; ii < folders; ii++)
		{
			m_xIntervals.insert(m_xIntervals.end(), tmpVct.begin(), tmpVct.end());
		}
		m_xIntervals.push_back(axisEnd);
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
