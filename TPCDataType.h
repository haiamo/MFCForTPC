/*
2019/6/12 TonyHe Add file and Vec3,Vec4 classes.
*/
#pragma once
#include <string>
#include <fstream>
#include <stdarg.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "cuda.h"
#include "device_functions.h"
#include "cublas.h"
#include "cublas_v2.h"
#include "stdio.h"
#include "curand.h"

using namespace std;

class TPCVec4
{
public:
	__device__ TPCVec4() :x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
	__device__ TPCVec4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
	//__device__ TPCVec4(const TPCVec4& inVec);
	__device__ ~TPCVec4();

	//__device__ TPCVec4& operator=(const TPCVec4 &inVec4);

	/*__device__ TPCVec4& operator+(const TPCVec4 &inVec4);

	__device__ TPCVec4& operator+=(const TPCVec4 & inVec4);

	__device__ TPCVec4& operator-(const TPCVec4 & inVec4);

	__device__ TPCVec4& operator-=(const TPCVec4 & inVec4);

	__device__ TPCVec4& operator*(const float &val);

	__device__ TPCVec4& operator*=(const float &val);

	__device__ TPCVec4& operator/(const float &val);

	__device__ TPCVec4& operator/=(const float &val);*/

	__device__ friend TPCVec4& operator+(const TPCVec4 &vec1, const TPCVec4 &vec2);

	__device__ friend TPCVec4& operator+=(TPCVec4 &vec1, const TPCVec4 &vec2);

	__device__ friend TPCVec4& operator-(const TPCVec4 &vec1, const TPCVec4 &vec2);

	__device__ friend TPCVec4& operator-=(TPCVec4 &vec1, const TPCVec4 &vec2);

	__device__ friend TPCVec4& operator*(const TPCVec4 &vec, const float &val);

	__device__ friend TPCVec4& operator*=(TPCVec4 &vec, const float &val);

	__device__ friend TPCVec4& operator/(const TPCVec4 &vec, const float &val);

	__device__ friend TPCVec4& operator/=(TPCVec4 &vec, const float &val);

public:
	float x;
	float y;
	float z;
	float w;
};
