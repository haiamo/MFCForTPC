#include "stdafx.h"
#include "TPCDataType.h"


/*__device__ TPCVec4::TPCVec4()
{
	x = 0.0f;
	y = 0.0f;
	z = 0.0f;
	w = 0.0f;
}

__device__ TPCVec4::TPCVec4(float _x, float _y, float _z, float _w)
{
	x = _x;
	y = _y;
	z = _z;
	w = _w;
}

__device__ TPCVec4::TPCVec4(const TPCVec4 & inVec)
{
	x = inVec.x;
	y = inVec.y;
	z = inVec.z;
	w = inVec.w;
}*/

__device__ TPCVec4::~TPCVec4()
{
}


/*__device__ TPCVec4& TPCVec4::operator=(const TPCVec4 & inVec4)
{
	if (this != &inVec4)
	{
		x = inVec4.x;
		y = inVec4.y;
		z = inVec4.z;
		w = inVec4.w;
	}
	return *this;
}

__device__ TPCVec4& TPCVec4::operator+(const TPCVec4 &inVec4)
{
	TPCVec4 tmp;
	tmp.x = x + inVec4.x;
	tmp.y = y + inVec4.y;
	tmp.z = z + inVec4.z;
	tmp.w = w + inVec4.w;
	return tmp;
}

__device__ TPCVec4& TPCVec4::operator+=(const TPCVec4 & inVec4)
{
	x += inVec4.x;
	y += inVec4.y;
	z += inVec4.z;
	w += inVec4.w;
	return *this;
}

__device__ TPCVec4 & TPCVec4::operator-(const TPCVec4 & inVec4)
{
	TPCVec4 tmp;
	tmp.x = x - inVec4.x;
	tmp.y = y - inVec4.y;
	tmp.z = z - inVec4.z;
	tmp.w = w - inVec4.w;
	return tmp;
}

__device__ TPCVec4& TPCVec4::operator-=(const TPCVec4 & inVec4)
{
	x -= inVec4.x;
	y -= inVec4.y;
	z -= inVec4.z;
	w -= inVec4.w;
	return *this;
}

//template<typename T>
__device__ TPCVec4 & TPCVec4::operator*(const float & val)
{
	TPCVec4 tmp;
	tmp.x = x * val;
	tmp.y = y * val;
	tmp.z = z * val;
	tmp.w = w * val;
	return tmp;
}

//template<typename T>
__device__ TPCVec4 & TPCVec4::operator*=(const float & val)
{
	x *= val;
	y *= val;
	z *= val;
	w *= val;
	return *this;
}

//template<typename T>
__device__ TPCVec4 & TPCVec4::operator/(const float & val)
{
	TPCVec4 tmp;
	tmp.x = x / val;
	tmp.y = y / val;
	tmp.z = z / val;
	tmp.w = w / val;
	return tmp;
}

__device__ TPCVec4 & TPCVec4::operator/=(const float & val)
{
	x /= val;
	y /= val;
	z /= val;
	w /= val;
	return *this;
}*/

__device__ TPCVec4& operator+(const TPCVec4 &vec1, const TPCVec4 &vec2)
{
	TPCVec4 tmp;
	tmp.x = vec1.x + vec2.x;
	tmp.y = vec1.y + vec2.y;
	tmp.z = vec1.z + vec2.z;
	tmp.w = vec1.w + vec2.w;
	return tmp;
}

__device__ TPCVec4& operator+=(TPCVec4 &vec1,const TPCVec4 &vec2)
{
	vec1.x += vec2.x;
	vec1.y += vec2.y;
	vec1.z += vec2.z;
	vec1.w += vec2.w;
	return vec1;
}

__device__ TPCVec4& operator-(const TPCVec4 &vec1, const TPCVec4 &vec2)
{
	TPCVec4 tmp;
	tmp.x = vec1.x - vec2.x;
	tmp.y = vec1.y - vec2.y;
	tmp.z = vec1.z - vec2.z;
	tmp.w = vec1.w - vec2.w;
	return tmp;
}

__device__ TPCVec4& operator-=(TPCVec4 &vec1, const TPCVec4 &vec2)
{
	vec1.x -= vec2.x;
	vec1.y -= vec2.y;
	vec1.z -= vec2.z;
	vec1.w -= vec2.w;
	return vec1;
}

__device__ TPCVec4& operator*(const TPCVec4 &vec, const float &val)
{
	TPCVec4 tmp;
	tmp.x = vec.x * val;
	tmp.y = vec.y * val;
	tmp.z = vec.z * val;
	tmp.w = vec.w * val;
	return tmp;
}

__device__ TPCVec4& operator*=(TPCVec4 &vec, const float &val)
{
	vec.x *= val;
	vec.y *= val;
	vec.z *= val;
	vec.w *= val;
	return vec;
}

__device__ TPCVec4& operator/(const TPCVec4 &vec, const float &val)
{
	TPCVec4 tmp;
	tmp.x = vec.x / val;
	tmp.y = vec.y / val;
	tmp.z = vec.z / val;
	tmp.w = vec.w / val;
	return tmp;
}

__device__ TPCVec4& operator/=(TPCVec4 &vec, const float &val)
{
	vec.x /= val;
	vec.y /= val;
	vec.z /= val;
	vec.w /= val;
	return vec;
}
