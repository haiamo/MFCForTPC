/**********
2019.3.6 Added TPCProperty class
This class is used for setting such properties that is needed in any procedure of TyrePointCloud.
***********/
#pragma once
#include <string>
#include <fstream>
#include <vector>

using namespace std;

struct LoadProp
{
	float begX;
	float endX;
	float begY;
	float endY;
	float begZ;
	float endZ;
	float stepX;
	float stepY;
	float stepZ;
	float originX;
	float originY;
	float originZ;
	string typeRange;
	string typeIntensity;
};

class TPCProperty
{
public:
	TPCProperty();
	virtual ~TPCProperty();

	int SetTPCLoadProp(string filename);

	int GetAxisBoundary(float* lb, float* ub, char axisName);

	int GetAxisProp(float* lb, float* ub, float* step, float* origin, char axisName);

	int SetAxisProp(float lb, float ub, float step, float origin, char axisName);

	int GetWidthHeightBSize(size_t* w, size_t* h, int *bs);

	int GetRIType(string& typeR, string& typeI);

	TPCProperty& operator=(const TPCProperty& prop);

private:
	size_t m_width;
	size_t m_height;
	int m_byteSize;
	LoadProp m_loadprop;
};

