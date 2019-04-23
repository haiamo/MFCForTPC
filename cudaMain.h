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

#ifdef __cplusplus

extern "C" cudaError_t RANSACOnGPU(double* xvals, double* yvals, size_t pcsize, int maxIters, int minInliers, int parasize, double uTh, double lTh,
							double* &paraList, int* &resInliers,	double* &modelErr, double* &dists, int &resIters);

extern "C" cudaError_t RANSACOnGPU1(double* xvals, double* yvals, size_t pcSize, int maxIters, int minInliers, int paraSize, double uTh, double lTh,
							double* &bestParas, int* &resInliers, double &modelErr, double* &dists);

extern "C" cudaError_t DataFitToGivenModel(double* xvals, double* yvals, size_t pcSize, int paraSize, double* modelPara,double uTh, double lTh,
							int &resInliers, double &modelErr, double* &dists);

#endif

#endif