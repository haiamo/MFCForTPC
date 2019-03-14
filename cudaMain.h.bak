/*	2019/2/28 TonyHE Verstion 1.0 of RANSAC method on GPU with cuda and cublas
1. The cuda main function is called RANSACOnGPU
2. CUDA kernel functions include SetupMatrices, GetModelPara and GetModelSqDist
*/

#pragma once
#include<time.h>//时间相关头文件，可用其中函数计算图像处理速度  
#include <stdio.h>
#include <iostream>
#include <string>

#include <pcl\point_types.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\transform.h>
#include <time.h>

#define datasize 500000

#ifndef INCLUDES_CUDAMAIN_H_
#define INCLUDES_CUDAMAIN_H_

#ifdef __cplusplus

extern "C" cudaError_t RANSACOnGPU(double* xvals, double* yvals, size_t pcsize, int maxIters, int minInliers, int parasize, double uTh, double lTh,
							double* &hst_hypox, double*& hst_hypoy, double* &As, double** &Qs, double** &taus, double** &Rs, double** &paras,
							double* &modelErr, double** &dists, int &hypoIters);

#endif

#endif