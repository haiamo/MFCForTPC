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

#define datasize 500000

#ifndef INCLUDES_CUDAMAIN_H_
#define INCLUDES_CUDAMAIN_H_

typedef struct
{
	double xbeg;
	double xend;
	double ybeg;
	double yend;
	double xstep;
	double ystep;
}CtrPtBound,GridProp;

#ifdef __cplusplus

extern "C" cudaError_t RANSACOnGPU(__in double* xvals,__in double* yvals,__in size_t pcsize,__in int maxIters,__in int minInliers,__in int parasize,__in double uTh,__in double lTh,
	__out double* &paraList, __out int* &resInliers, __out	double* &modelErr, __out double* &dists, __out int &resIters);

extern "C" cudaError_t RANSACOnGPU1(__in double* xvals, __in double* yvals, __in size_t pcSize, __in int maxIters, __in int minInliers, __in int paraSize, __in double uTh, __in double lTh,
	__out double* &bestParas, __out int* &resInliers, __out double &modelErr, __out double* &dists);

extern "C" cudaError_t DataFitToGivenModel(__in double* xvals, __in double* yvals, __in size_t pcSize, __in int paraSize, __in double* modelPara, __in double uTh, __in double lTh,
	__out int &resInliers, __out double &modelErr, __out double* &dists);

extern "C" cudaError_t NURBSRANSACOnGPU(__in CtrPtBound inBound, __in double* xvals, __in double* yvals, __in size_t pcsize, __in int maxIters, __in int minInliers, __in double Threshold,
	__out int*& resInliers, __out double &modelErr, __out double* &bestCtrx, __out double* &bestCtry);

extern "C" cudaError_t GenNURBSCurve(__in GridProp inGP, __in double* ctrPtx, __in double* ctrPty, __in int ctrSize, __in size_t pcSize, __inout double* xvals, __inout double* yvals);
#endif

#endif