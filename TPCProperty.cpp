#include "stdafx.h"
#include "TPCProperty.h"


TPCProperty::TPCProperty()
{
	m_width = 0;
	m_height = 0;
	m_byteSize = 0;
	m_loadprop.begX = 0.0;
	m_loadprop.begY = 0.0;
	m_loadprop.begZ = 0.0;
	m_loadprop.endX = 0.0;
	m_loadprop.endY = 0.0;
	m_loadprop.endZ = 0.0;
	m_loadprop.stepX = 0.0;
	m_loadprop.stepY = 0.0;
	m_loadprop.stepZ = 0.0;
	m_loadprop.originX = 0.0;
	m_loadprop.originY = 0.0;
	m_loadprop.originZ = 0.0;
	m_loadprop.typeRange = "";
	m_loadprop.typeIntensity = "";
}


TPCProperty::~TPCProperty()
{
}

int TPCProperty::SetTPCLoadProp(string filename)
{
	fstream file;
	file.open(filename, ios_base::in);
	string strLine,strVal;
	float origin,scale;
	string::size_type idx, begIdx, endIdx;
	bool b_sensor = false, b_world = false, b_range = false, b_intensity = false, b_databegin = false;
	vector<string> baseStrs = { "fov x0","fov x1","fov z0","fov z2","origin x","scale x","origin z","scale z"};
	vector<string>::iterator bStrIt = baseStrs.begin();
	vector<string> typeStrs = { "BYTE","WORD","DWORD","INT","FLOAT" };

	while (getline(file,strLine))
	{
		if (!b_databegin)
		{
			if (string::npos != strLine.find("<sensorrangetraits>"))
			{
				b_sensor = true;
				b_databegin = true;
				continue;
			}

			if (string::npos != strLine.find("<worldrangetraits>"))
			{
				b_world = true;
				b_databegin = true;
				continue;
			}

			if (string::npos != strLine.find("\"Range\"") && string::npos!=strLine.find("<subcomponent"))
			{
				b_range = true;
				b_databegin = true;
				for (vector<string>::iterator tmpIt = typeStrs.begin(); tmpIt < typeStrs.end(); tmpIt++)
				{
					if (string::npos != strLine.find(*tmpIt))
					{
						m_loadprop.typeRange = *tmpIt;
						break;
					}
				}
				continue;
			}
			
			if(string::npos!= strLine.find("\"Intensity\"") && string::npos != strLine.find("<subcomponent"))
			{
				b_intensity = true;
				b_databegin = true;
				for (vector<string>::iterator tmpIt = typeStrs.begin(); tmpIt < typeStrs.end(); tmpIt++)
				{
					if (string::npos != strLine.find(*tmpIt))
					{
						m_loadprop.typeIntensity = *tmpIt;
						break;
					}
				}
				continue;
			}
		}
		else
		{

			if (string::npos != strLine.find("</sensorrangetraits>"))
			{
				b_databegin = false;
			}

			if (string::npos != strLine.find("</worldrangetraits>"))
			{
				b_databegin = false;
			}

			if (string::npos != strLine.find("</subcomponent>") && b_range)
			{
				b_databegin = false;
			}
			
			if (b_sensor)
			{
				if (bStrIt != baseStrs.end())
				{
					idx = strLine.find(*bStrIt);
					if (string::npos != idx)
					{
						begIdx = strLine.find(">");
						endIdx = strLine.find("<", begIdx);
						strVal = strLine.substr(begIdx + 1, endIdx - begIdx - 1);
						switch (bStrIt - baseStrs.begin())
						{
						case 0:
							m_loadprop.begX = stof(strVal);
							break;
						case 1:
							m_loadprop.endX = stof(strVal);
							break;
						case 2:
							m_loadprop.begZ = stof(strVal);
							break;
						case 3:
							m_loadprop.endZ = stof(strVal);
							break;
						case 4:
							origin = stof(strVal);
							m_loadprop.originX = origin;
							break;
						case 5:
							scale = stof(strVal);
							//m_loadprop.endX = (m_loadprop.endX - m_loadprop.begX)*scale + origin;
							//m_loadprop.begX = origin;
							m_loadprop.stepX = abs(scale);
							m_loadprop.stepY = abs(scale);
							break;
						case 6:
							origin = stof(strVal);
							m_loadprop.originZ = origin;
							break;
						case 7:
							scale = stof(strVal);
							//m_loadprop.endZ = (m_loadprop.endZ - m_loadprop.begZ)*scale + origin;
							//m_loadprop.begZ = origin;
							m_loadprop.stepZ = scale;
							//if (abs(scale) < m_loadprop.stepY)
							//{
								//m_loadprop.stepY = abs(scale);
								//m_loadprop.stepX = abs(scale);
							//}
							break;
						}
						bStrIt++;
					}
				}
				else
				{
					b_sensor = false;
				}
				
			}

			if (b_range)
			{
				idx = strLine.find("width");
				if (string::npos != idx)
				{
					begIdx = idx + 7;
					endIdx = strLine.find("<",idx);
					strVal = strLine.substr(begIdx, endIdx - begIdx);
					m_width = stoi(strVal);
				}
			}
		}
	} 

	/*float tmpVal = 0.0;
	if (m_loadprop.begX > m_loadprop.endX)
	{
		tmpVal = m_loadprop.begX;
		m_loadprop.begX = m_loadprop.endX;
		m_loadprop.endX = tmpVal;
		m_loadprop.stepX = abs(m_loadprop.stepX);
	}

	if (m_loadprop.begY > m_loadprop.endY)
	{
		tmpVal = m_loadprop.begY;
		m_loadprop.begY = m_loadprop.endY;
		m_loadprop.endY = tmpVal;
		m_loadprop.stepY = abs(m_loadprop.stepY);
	}

	if (m_loadprop.begZ > m_loadprop.endZ)
	{
		tmpVal = m_loadprop.begZ;
		m_loadprop.begZ = m_loadprop.endZ;
		m_loadprop.endZ = tmpVal;
		m_loadprop.stepZ = abs(m_loadprop.stepZ);
	}*/

	m_byteSize = 0;
	string curStr = m_loadprop.typeRange;
	if (typeStrs[0] == curStr)
	{
		m_byteSize += 1;
	}
	else if (typeStrs[1] == curStr)
	{
		m_byteSize += 2;
	}
	else if (typeStrs[2] == curStr || typeStrs[3] == curStr || typeStrs[4] == curStr)
	{
		m_byteSize += 4;
	}

	curStr = m_loadprop.typeIntensity;
	if (typeStrs[0] == curStr)
	{
		m_byteSize += 1;
	}
	else if (typeStrs[1] == curStr)
	{
		m_byteSize += 2;
	}
	else if (typeStrs[2] == curStr || typeStrs[3] == curStr || typeStrs[4] == curStr)
	{
		m_byteSize += 4;
	}
	return 0;
}

int TPCProperty::GetAxisBoundary(float * lb, float * ub, char axisName)
{
	float beg, end, step, origin;
	switch (axisName)
	{
	case 'x':
	case 'X':
		beg = m_loadprop.begX;
		end = m_loadprop.endX;
		step = m_loadprop.stepX;
		origin = m_loadprop.originX;
		break;
	case 'y':
	case 'Y':
		beg = m_loadprop.begY;
		end = m_loadprop.endY;
		step = m_loadprop.stepY;
		origin = m_loadprop.originY;
		break;
	case 'z':
	case 'Z':
		beg = m_loadprop.begZ;
		end = m_loadprop.endZ;
		step = m_loadprop.stepZ;
		origin = m_loadprop.originZ;
		break;
	}
	if (step > 0)
	{
		*lb = origin;
		*ub = origin + (end - beg)*step;
	}
	else
	{
		*ub = origin;
		*lb = origin + (end - beg)*step;
	}
	return 0;
}

int TPCProperty::GetAxisProp(float * lb, float * ub, float * step, float* origin, char axisName)
{
	switch (axisName)
	{
	case 'x':
	case 'X':
		if (NULL != lb)
		{
			*lb = m_loadprop.begX;
		}
			
		if (NULL != ub)
		{
			*ub = m_loadprop.endX;
		}
			
		if (NULL != step)
		{
			*step = m_loadprop.stepX;
		}
		
		if (NULL != origin)
		{
			*origin = m_loadprop.originX;
		}
		break;
	case 'y':
	case 'Y':
		if (NULL != lb)
		{
			*lb = m_loadprop.begY;
		}

		if (NULL != ub)
		{
			*ub = m_loadprop.endY;
		}

		if (NULL != step)
		{
			*step = m_loadprop.stepY;
		}

		if (NULL != origin)
		{
			*origin = m_loadprop.originY;
		}
		break;
	case 'z':
	case 'Z':
		if (NULL != lb)
		{
			*lb = m_loadprop.begZ;
		}

		if (NULL != ub)
		{
			*ub = m_loadprop.endZ;
		}

		if (NULL != step)
		{
			*step = m_loadprop.stepZ;
		}

		if (NULL != origin)
		{
			*origin = m_loadprop.originZ;
		}
		break;
	}
	return 0;
}

int TPCProperty::SetAxisProp(float lb, float ub, float step, float origin, char axisName)
{
	switch (axisName)
	{
	case 'x':
	case 'X':
		m_loadprop.begX = lb;
		m_loadprop.endX = ub;
		m_loadprop.originX = origin;
		m_loadprop.stepX = step;
		break;
	case 'y':
	case 'Y':
		m_loadprop.begY = lb;
		m_loadprop.endY = ub;
		m_loadprop.originY = origin;
		m_loadprop.stepY = step;
		break;
	case 'z':
	case 'Z':
		m_loadprop.begZ = lb;
		m_loadprop.endZ = ub;
		m_loadprop.originZ = origin;
		m_loadprop.stepZ = step;
		break;
	}
	return 0;
}

int TPCProperty::GetWidthHeightBSize(size_t * w, size_t * h, int* bs)
{
	if (NULL != w)
	{
		*w = m_width;
	}

	if (NULL != h)
	{
		*h = m_height;
	}

	if (NULL != bs)
	{
		*bs = m_byteSize;
	}
	return 0;
}

int TPCProperty::GetRIType(string & typeR, string & typeI)
{
	typeR = m_loadprop.typeRange;
	typeI = m_loadprop.typeIntensity;
	return 0;
}

TPCProperty & TPCProperty::operator=(const TPCProperty & prop)
{
	if (this != &prop)
	{
		this->m_byteSize = prop.m_byteSize;
		this->m_height = prop.m_height;
		this->m_width = prop.m_width;
		this->m_loadprop.begX = prop.m_loadprop.begX;
		this->m_loadprop.begY = prop.m_loadprop.begY;
		this->m_loadprop.begZ = prop.m_loadprop.begZ;
		this->m_loadprop.endX = prop.m_loadprop.endX;
		this->m_loadprop.endY = prop.m_loadprop.endY;
		this->m_loadprop.endZ = prop.m_loadprop.endZ;
		this->m_loadprop.originX = prop.m_loadprop.originX;
		this->m_loadprop.originY = prop.m_loadprop.originY;
		this->m_loadprop.originZ = prop.m_loadprop.originZ;
		this->m_loadprop.stepX = prop.m_loadprop.stepX;
		this->m_loadprop.stepY = prop.m_loadprop.stepY;
		this->m_loadprop.stepZ = prop.m_loadprop.stepZ;
		this->m_loadprop.typeIntensity = prop.m_loadprop.typeIntensity;
		this->m_loadprop.typeRange = prop.m_loadprop.typeRange;
	}
	return *this;
}
