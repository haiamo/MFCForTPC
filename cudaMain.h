/*	2019/2/28 TonyHE Verstion 1.0 of RANSAC method on GPU with cuda and cublas
1. The cuda main function is called RANSACOnGPU
2. CUDA kernel functions include SetupMatrices, GetModelPara and GetModelSqDist
*/

#pragma once
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <string>

#include <pcl\point_types.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\transform.h>
#include <time.h>

#include "FileLogger.h"
//#include "TPCDataType.h"
#include "Utility.h"

#define datasize 500000

#ifndef INCLUDES_CUDAMAIN_H_
#define INCLUDES_CUDAMAIN_H_

#define USERDEBUG 1

#define CHECK_D_ZERO(x) (((x)<1e-6) && ((x)>-1e-6))
#define THREAD_SIZE 32

#define GETGRIDTOTAL(dtLen) ((dtLen)+THREAD_SIZE*THREAD_SIZE-1)/THREAD_SIZE/THREAD_SIZE
#define GETGRIDX(dtLen)  sqrtl(((dtLen) + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE)
#define GETGRIDY(dtLen, x)  (((dtLen) + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE + (x) - 1) / (x)

typedef struct
{
	double xbeg;
	double xend;
	double ybeg;
	double yend;
	double xstep;
	double ystep;
}CtrPtBound,GridProp;

typedef union __align__(16)
{
	struct {
		float x3;
		float y3;
		float z3;
		unsigned int id;
	};

	struct {
		float x2;
		float y2;
		float w2;
	};
}Point3D, Point2Dw;

typedef struct __align__(8) {
	float x;
	float y;
}Point2D;

class TPCVec4
{
public:
	__host__ __device__ TPCVec4() :x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
	__host__ __device__ TPCVec4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {}
	__host__ __device__ TPCVec4(const Point3D& inVec, const float wt);
	__host__ __device__ ~TPCVec4();

	__device__ TPCVec4 HomoCoordPtToPt3Dw();

	__device__ TPCVec4 Pt3DwToHomoCoordPt();

	__device__ Point3D HomoCoordPtToPt3D();

	__device__ TPCVec4 Pt3DToHomoCoordPt(Point3D inV, float wt);

	__device__ void GetPoint3D(Point3D& outP3D);

	__device__ TPCVec4 Cross(TPCVec4 v);

	__device__ double Dot(TPCVec4 v);

	__device__ double Length();

	__host__ __device__ TPCVec4 operator=(const TPCVec4 &inVec4);

	__device__ friend TPCVec4 operator+(const TPCVec4 &vec1, const TPCVec4 &vec2);

	__device__ friend TPCVec4 operator+=(TPCVec4 &vec1, const TPCVec4 &vec2);

	__device__ friend TPCVec4 operator-(const TPCVec4 &vec1, const TPCVec4 &vec2);

	__device__ friend TPCVec4 operator-=(TPCVec4 &vec1, const TPCVec4 &vec2);

	__device__ friend TPCVec4 operator*(const TPCVec4 &vec, const float &val);

	__device__ friend TPCVec4 operator*(const float &val, const TPCVec4 &vec);

	__device__ friend TPCVec4 operator*=(TPCVec4 &vec, const float &val);

	__device__ friend TPCVec4 operator/(const TPCVec4 &vec, const float &val);

	__device__ friend TPCVec4 operator/=(TPCVec4 &vec, const float &val);

public:
	float x;
	float y;
	float z;
	float w;
};

typedef TPCVec4 Point3Dw;
typedef TPCVec4 PointQuat;

__device__ bool dIsZero(double inVal);

int FreeCudaPtrs(int cnt, ...);

template<class T>
void GetGridFactor(T totalNum, T blockSize, T& outNum1, T& outNum2);

__device__ void FindSpan(__in int n, __in int p, __in double u, __in double* U, __out int* span);

__device__ void BasisFuns(__in int i, __in double u, __in int p, __in double* U, __in double* left, __in double* right, __out double* N);

__device__ void DersBasisFuns(__in int i, __in int p, __in double u, __in int nd, __in double* U,
	__in double* ndu, __in double* a, __in double* left, __in double* right, __out double* ders);

__global__ void GetControlPointsFromDataPointP3D(__in Point3D* dataPt, __in double* dataH, __in unsigned int dataSize,
	__in Point3Dw* P, __in double* A, __in double* B, __in double* C, __in double* _C, __in Point3Dw* E, __in Point3Dw* _E,
	__in double* U,__out Point3Dw* ctrPt);

#ifdef __cplusplus

extern "C" cudaError_t RANSACOnGPU(__in double* xvals,__in double* yvals,__in size_t pcsize,__in int maxIters,__in int minInliers,__in int parasize,__in double uTh,__in double lTh,
	__out double* &paraList, __out int* &resInliers, __out	double* &modelErr, __out double* &dists, __out int &resIters);

extern "C" cudaError_t RANSACOnGPU1(__in double* xvals, __in double* yvals, __in size_t pcSize, __in int maxIters, __in int minInliers, __in int paraSize, __in double uTh, __in double lTh,
	__out double* &bestParas, __out int* &resInliers, __out double &modelErr, __out double* &dists);

extern "C" cudaError_t DataFitToGivenModel(__in double* xvals, __in double* yvals, __in size_t pcSize, __in int paraSize, __in double* modelPara, __in double uTh, __in double lTh,
	__out int &resInliers, __out double &modelErr, __out double* &dists);

extern "C" cudaError_t NURBSRANSACOnGPU(__in CtrPtBound inBound, __in double* xvals, __in double* yvals, __in size_t pcsize, __in int maxIters, __in int minInliers, __in double UTh, __in double LTh,
	__out int*& resInliers, __out double &modelErr, __out double* &bestCtrx, __out double* &bestCtry, __out double*&bestDists);

extern "C" const char* NURBSRANSACOnGPUStream(__in CtrPtBound inBound, __in double* xvals, __in double* yvals, __in size_t pcsize, __in int maxIters, __in int minInliers, __in double UTh, __in double LTh,
	__out int*& resInliers, __out double &modelErr, __out double* &bestCtrx, __out double* &bestCtry, __out double*&bestDists);


extern "C" cudaError_t GenNURBSCurve(__in double* dataPtx, __in double* dataPty,__in double* H, __in int dataSize, __in size_t pcSize, __inout double* xvals, __inout double* yvals);

extern "C" cudaError_t GenNURBSCurveByCtr(__in Point3Dw *ctrPts, __in unsigned int ctrSize, __in unsigned int dataSize, __out double* xvals, __out double* zvals);

extern "C" const char* NURBSCurveSearchGPU(__in GridProp ptProp, __in Point3D* inPts,__in unsigned int ptSize, __in char lineName, __in float cvThrs,__in unsigned int cvSize, __in float dgThrs, __in unsigned int nghbs, __out int* outIDs, __out Point3Dw* outCtrs, Point3D* outCvPts);
#endif

#endif