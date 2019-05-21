#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "cuda.h"
#include "device_functions.h"
#include "cublas.h"
#include "cublas_v2.h"
#include "cudaMain.h"
#include "stdio.h"
#include "curand.h"
#include "FileLogger.h"

#define USERDEBUG 0

#define CHECK_D_ZERO(x) (abs(x))<1e-10
#define THREAD_SIZE 16

#define GETGRIDTOTAL(dtLen) ((dtLen)+THREAD_SIZE*THREAD_SIZE-1)/THREAD_SIZE/THREAD_SIZE
#define GETGRIDX(dtLen)  sqrtl(((dtLen) + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE)
#define GETGRIDY(dtLen, x)  (((dtLen) + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE + (x) - 1) / (x)

__host__ cudaError_t UserDebugCopy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind)
{
	cudaError_t res = cudaSuccess;
	if(USERDEBUG)
	{
		res = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
	}
	return res;
}

__host__ cudaError_t UserDebugCopy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
	cudaError_t res = cudaSuccess;
	if (USERDEBUG)
	{
		res = cudaMemcpy(dst, src, count, kind);
	}
	return res;
}

__global__ void SetupMatrices(double* xvals, int maxIt, cudaExtent Aextent, cudaPitchedPtr Amat)
{
	int bID = blockIdx.x + blockIdx.y*gridDim.x;
	int tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tID < maxIt)
	{
		double* curA = (double*)((char*)Amat.ptr + tID*Amat.pitch*Aextent.height);
		int curID = 0, preLineID = 0;
		int Arows = Aextent.width / sizeof(double), Acols = Aextent.height, ApitchLen = Amat.pitch / sizeof(double);
		for (int coli = 0; coli < Acols; coli++)
		{
			for (int rowi = 0; rowi < Arows; rowi++)
			{
				curID = coli*ApitchLen + rowi;
				if (coli == 0)
				{
					curA[curID] = 1;
				}
				else
				{
					preLineID = (coli - 1)*ApitchLen + rowi;
					curA[curID] = curA[preLineID] * xvals[tID*Arows + rowi];
				}
			}
		}
	}
}

__global__ void GetHypoModelPara(double* yvals,int maxIt, size_t pcSize, cudaPitchedPtr Amat, cudaExtent Aextent,
	size_t colvPitch, double* colVecs, size_t paraPitch, double* paras)
{
	int bID = blockIdx.x + blockIdx.y*gridDim.x;
	int tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tID < maxIt)
	{
		double* curA = (double*)((char*)Amat.ptr + tID*Amat.pitch*Aextent.height);//Column major storage
		double* curYs = yvals + tID*pcSize;
		double *outPara = (double*)((char*)paras + tID*paraPitch), tmpVal = 0.0;

		int curID = 0;
		double vnorm = 0.0, sigma = 0.0, beta = 0.0, ytv = 0.0;
		int Arows = Aextent.width / sizeof(double), Acols = Aextent.height, ApitchLen = Amat.pitch / sizeof(double);
		double* col_vec = (double*)((char*)colVecs + tID*colvPitch), *Atv_vec = (double*)malloc(sizeof(double)*Acols);
		
		for (int coli = 0; coli < Acols; coli++)
		{
			outPara[coli] = 0.0;
			Atv_vec[coli] = 0.0;
		}

		//QR decomposition by Householder reflection.
		for (int coli = 0; coli < Acols; coli++)
		{
			//Compute column vector(col_vec) norm
			vnorm = 0.0;
			for (int rowi = coli; rowi < Arows; rowi++)
			{
				curID = coli*ApitchLen + rowi;
				vnorm += curA[curID] * curA[curID];
				col_vec[rowi] = curA[curID];
			}
			vnorm = sqrt(vnorm);

			//Compute beta
			if (curA[coli*ApitchLen + coli] < 0)
			{
				sigma = -vnorm;
			}
			else
			{
				sigma = vnorm;
			}
			col_vec[coli] += sigma;
			beta = 1 / sigma / (sigma + curA[coli*ApitchLen + coli]);

			//Compute A^t*col_vec and y^t*col_vec
			for (int colj = coli; colj < Acols; colj++)
			{
				Atv_vec[colj] = 0.0;
				if (colj == coli)
				{
					ytv = 0.0;
				}
				for (int rowj = coli; rowj < Arows; rowj++)
				{
					curID = colj*ApitchLen + rowj;
					Atv_vec[colj] += curA[curID] * col_vec[rowj];
					if (CHECK_D_ZERO(Atv_vec[colj]))
					{
						Atv_vec[colj] = 0.0;
					}
					if (colj == coli)
					{
						ytv += curYs[rowj] * col_vec[rowj];
					}
				}
			}

			//H(k)A(k)=A(k-1)-beta(k-1)*col_vec(k-1)*Atv_vec^t(k-1)
			//y(k)=y(k-1)-bata(k-1)*col_vec(k-1)*ytv(k-1)
			for (int colj = coli; colj < Acols; colj++)
			{
				for (int rowj = coli; rowj < Arows; rowj++)
				{
					curID = colj*ApitchLen + rowj;
					curA[curID] -= beta*col_vec[rowj] * Atv_vec[colj];
					if (CHECK_D_ZERO(curA[curID]))
					{
						curA[curID] = 0.0;
					}
					if (colj == coli)
					{
						curYs[rowj] -= beta*col_vec[rowj] * ytv;
					}
				}
			}
		}
		//Now, A->QA=R, y->Qy; Aalpha=y->Ralpha=Qy, the next step will obtain alpha=R^(-1)Qy
		for (int rowi = Acols - 1; rowi >= 0; rowi--)
		{
			tmpVal = curYs[rowi];
			//Calculate the ii-th parameter in alpha
			for (int coli = rowi + 1; coli < Acols; coli++)
			{
				curID = coli*ApitchLen + rowi;
				tmpVal -= outPara[coli] * curA[curID];
			}
			outPara[rowi] = tmpVal / curA[rowi*ApitchLen + rowi];
		}
		free(Atv_vec);
		Atv_vec = NULL;
	}
}

__global__ void CheckPointInOrOut(double* xs, double* ys, int maxIt, size_t pcSize, int paraSize, double uTh,double lTh, size_t paraPitch, double* paras,
	int* inlierNum, size_t bInOutPitch, bool* bInOut, double* modelErr,size_t distPitch, double* dists)
{
	int bID = blockIdx.x + blockIdx.y*gridDim.x;
	int tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tID < maxIt)
	{
		double* curPara = (double*)((char*)paras+ tID*paraPitch), *outErr = modelErr + tID, *outDist = (double*)((char*)dists + tID*distPitch);
		*outErr = 0.0;
		bool *curbInOut = (bool*)((char*)bInOut + tID*bInOutPitch);
		int *curInlNum = inlierNum + tID;
		*curInlNum = 0;
		double yHat = 0.0;//y^^=paras[k]*x^(k-1)+...paras[0], k=order of fitted polynomial
		double xPows = 0.0, tmpDist = 0.0, tmpErr = 0.0;
		for (size_t ii = 0; ii < pcSize; ii++)
		{
			xPows = 1.0;
			yHat = 0.0;
			curbInOut[ii] = false;
			for (int parii = 0; parii < paraSize; parii++)
			{
				yHat += xPows*curPara[parii];
				xPows *= xs[ii];
			}
			tmpDist = ys[ii] - yHat;
			outDist[ii] = tmpDist;
			tmpErr += tmpDist*tmpDist;
			if ((tmpDist > 0 && tmpDist - uTh<0.0) || (tmpDist < 0 && tmpDist + lTh > 0.0))
			{
				(*curInlNum)++;
				curbInOut[ii] = true;
			}
		}
		tmpErr /= pcSize;
		*outErr = tmpErr;
	}
}

__global__ void SetupMatriceswithVariousSizes(double* xvals, int maxIt, int pcSize, int paraSize,
	int* inlierNum,size_t bInOutPitch, bool*bInOrOut, cudaExtent Aextent, cudaPitchedPtr Amat)
{
	int bID = blockIdx.x + blockIdx.y*gridDim.x;
	int tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tID < maxIt)
	{
		int curInliers = *(inlierNum + tID);
		if (curInliers > paraSize)
		{
			bool* curInOrOut = (bool*)((char*)bInOrOut + tID*bInOutPitch);
			double* curA = (double*)((char*)Amat.ptr + tID*Amat.pitch*Aextent.height);
			int Arows = curInliers, Acols = Aextent.height, ApitchLen = Amat.pitch / sizeof(double);
			int curID = 0, preLineID = 0;

			int rowi = 0;
			for (int pcID = 0; pcID < pcSize; pcID++)
			{
				if (curInOrOut[pcID] && rowi < Arows)
				{
					for (int coli = 0; coli < Acols; coli++)
					{
						curID = coli*ApitchLen + rowi;
						if (coli == 0)
						{
							curA[curID] = 1;
						}
						else
						{
							preLineID = (coli - 1)*ApitchLen + rowi;
							curA[curID] = curA[preLineID] * xvals[pcID];
						}
					}
					rowi++;
				}
			}
		}
	}
}

__global__ void GetModelParawithVariateAs(double* yvals,int maxIt, size_t pcSize,int paraSize, cudaExtent Aextent, cudaPitchedPtr Amat, int* inlierNum,
	  size_t colVPitch, double* colVecs,size_t Atv_pitch, double* Atvs, size_t allYsPitch, double* allYs, size_t bInOutPitch, bool* bInOut, size_t paraPitch, double* paras)
{
	//This function uses QR decomposition to get the least-square model for each iteration. The size of input matrices A can be various.
	int bID = blockIdx.x + blockIdx.y*gridDim.x;
	int tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tID < maxIt)
	{
		int Arows = *(inlierNum + tID), Acols = paraSize;
		if (Arows > Acols)
		{
			bool* curbInlier = (bool*)((char*)bInOut + tID*bInOutPitch);
			double* curA = (double*)((char*)Amat.ptr + tID*Amat.pitch*Aextent.height);
			double *outPara = (double*)((char*)paras+ tID*paraPitch), tmpVal = 0.0;

			int curID = 0;
			double vnorm = 0.0, sigma = 0.0, beta = 0.0, ytv = 0.0;
			double* col_vec = (double*)((char*)colVecs + tID*colVPitch), *Atv_vec = (double*)((char*)Atvs+ tID*Atv_pitch), *curYs = (double*)((char*)allYs + tID*allYsPitch);
			size_t ApitchLen = Amat.pitch / sizeof(double);
			
			//Prepared the y column vector.
			for (size_t rowi = 0; rowi < pcSize; rowi++)
			{
				if (curbInlier[rowi])
				{
					curYs[curID] = yvals[rowi];
					curID++;
				}
			}

			for (int coli = 0; coli < Acols; coli++)
			{
				outPara[coli] = 0.0;
			}

			//QR decomposition by Householder reflection.
			for (int coli = 0; coli < Acols; coli++)
			{
				//Compute column vector(col_vec) norm
				vnorm = 0.0;
				for (int rowi = coli; rowi < Arows; rowi++)
				{
					curID = coli*ApitchLen + rowi;
					vnorm += curA[curID] * curA[curID];
					col_vec[rowi] = curA[curID];
				}
				vnorm = sqrt(vnorm);

				//Compute beta
				if (curA[coli*ApitchLen + coli] < 0)
				{
					sigma = -vnorm;
				}
				else
				{
					sigma = vnorm;
				}
				col_vec[coli] +=sigma;
				beta = 1 / sigma / (sigma + curA[coli*ApitchLen +coli]);

				//Compute A^t*col_vec and y^t*col_vec
				for (int colj = coli; colj < Acols; colj++)
				{
					Atv_vec[colj] = 0.0;
					if (colj == coli)
					{
						ytv = 0.0;
					}
					for (int rowj = coli; rowj < Arows; rowj++)
					{
						curID = colj*ApitchLen + rowj;
						Atv_vec[colj] += curA[curID] * col_vec[rowj];
						if (colj == coli)
						{
							ytv += curYs[rowj] * col_vec[rowj];
						}
					}
				}

				//H(k)A(k)=A(k-1)-beta(k-1)*col_vec(k-1)*Atv_vec^t(k-1)
				//y(k)=y(k-1)-bata(k-1)*col_vec(k-1)*ytv(k-1)
				for (int colj = coli; colj < Acols; colj++)
				{
					for (int rowj = coli; rowj < Arows; rowj++)
					{
						curID = colj*ApitchLen + rowj;
						curA[curID] -= beta*col_vec[rowj] * Atv_vec[colj];
						if (CHECK_D_ZERO(curA[curID]))
						{
							curA[curID] = 0.0;
						}
						if (colj == coli)
						{
							curYs[rowj] -= beta*col_vec[rowj] * ytv;
						}
					}
				}
			}
			//Now, A->Q*A=R, y->Q*y; A*alpha=y->R*alpha=Q*y, the next step will obtain alpha=R^(-1)*Q*y
			for (int rowi = Acols - 1; rowi >= 0; rowi--)
			{
				tmpVal = curYs[rowi];
				//Calculate the ii-th parameter in alpha
				for (int coli = rowi + 1; coli < Acols; coli++)
				{
					curID = coli*ApitchLen + rowi;
					tmpVal -= outPara[coli] * curA[curID];
				}
				outPara[rowi] = tmpVal / curA[rowi*ApitchLen + rowi];
			}
		}
	}
}

__global__ void GetModelSqDist(double* xs, double* ys,int maxIt,size_t pcSize,  int paraSize, size_t paraPitch, double* paras, int* inlierNum,
							 double* modelErr,size_t distPitch, double* dists)
{
	int bID = blockIdx.x + blockIdx.y*gridDim.x;
	int tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tID < maxIt)
	{
		int Arows = *(inlierNum + tID), Acols = paraSize;
		if (Arows > Acols)
		{
			double* curPara = (double*)((char*)paras + tID*paraPitch), *outErr = modelErr + tID, *outDist = (double*)((char*)dists + tID*distPitch);
			double yHat = 0.0;//y^^=paras[k]*x^(k-1)+...paras[0], k=order of fitted polynomial
			double xPows = 0.0, tmpDist = 0.0, tmpErr = 0.0;
			*outErr = 0.0;
			for (size_t ii = 0; ii < pcSize; ii++)
			{
				xPows = 1.0;
				yHat = 0.0;
				for (int parii = 0; parii < paraSize; parii++)
				{
					yHat += xPows*curPara[parii];
					xPows *= xs[ii];
				}
				tmpDist = ys[ii] - yHat;
				outDist[ii] = tmpDist;
				tmpErr += tmpDist*tmpDist;
			}
			tmpErr /= pcSize;
			*outErr = tmpErr;
		}
	}
}

extern "C" cudaError_t RANSACOnGPU(__in double* xvals, __in double* yvals, __in size_t pcSize, __in int maxIters, __in int minInliers, __in int paraSize, __in double uTh, __in double lTh,
	__out double* &paraList, __out int* &resInliers, __out	double* &modelErr, __out double* &dists, __out int &resIters)
{
	//Set GPU device structure: grid and block.
	cudaError_t cudaErr;
	unsigned int gridX = (unsigned int)sqrtl((maxIters + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE);
	unsigned int gridY = (unsigned int)((maxIters + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE + gridX - 1) / gridX;
	dim3 blocks(THREAD_SIZE, THREAD_SIZE), grids(gridX, gridY);

	/***********
	PART I: This part is to figure out the fitted model by using hypo-inliers, which is a subset of the whole data points.

	Get the 2D curve model by Least Square Method with QR Decomposition
	Sove the equation A*a=y, where A has the size n*m, a has m*1 and y has n*1.
	In this problem, n(rows) is the size of hypo-inliers Points, m(cols) equals to the order of fitted ploynomial plus one(m=order+1).
	For parallel computing, there are MaxIteration(mi:num) matrices in total.
	All matrices is column-major stored continuously.

	************/
	double *d_hypox = NULL, *d_hypoy = NULL, *h_hypox = NULL, *h_hypoy = NULL;
	unsigned int *h_hypoIDs;
	h_hypoIDs = (unsigned int *)malloc(sizeof(unsigned int)*maxIters*minInliers);
	h_hypox = (double*)malloc(sizeof(double)*maxIters*minInliers);
	h_hypoy = (double*)malloc(sizeof(double)*maxIters*minInliers);

	//Step 1.1: Generate random hypo-inliers IDs and setup matrices for further algorithm.
	srand(unsigned(time(NULL)));
	for (size_t ii = 0; ii < maxIters*minInliers; ii++)
	{
		h_hypoIDs[ii] = rand() % pcSize;
		h_hypox[ii] = xvals[h_hypoIDs[ii]];
		h_hypoy[ii] = yvals[h_hypoIDs[ii]];
	}

	cudaErr = cudaMalloc((void**)&d_hypox, sizeof(double)*maxIters*minInliers);
	cudaErr = cudaMalloc((void**)&d_hypoy, sizeof(double)*maxIters*minInliers);

	cudaErr = cudaMemcpy(d_hypox, h_hypox, sizeof(double)*maxIters*minInliers, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_hypoy, h_hypoy, sizeof(double)*maxIters*minInliers, cudaMemcpyHostToDevice);

	cudaPitchedPtr d_Amat;
	cudaExtent Aextent;
	Aextent = make_cudaExtent(sizeof(double)*minInliers, paraSize, maxIters);
	cudaErr = cudaMalloc3D(&d_Amat, Aextent);
	SetupMatrices <<<grids, blocks>>> (d_hypox, maxIters, Aextent, d_Amat);
	//cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();

	//Step 1.2: Get model parameters by using hypo-inlier point data.
	double* d_ColVec, *d_Paras;
	size_t colVecPitch, parasPitch;
	cudaErr = cudaMallocPitch((void**)&d_ColVec, &colVecPitch, sizeof(double)*minInliers, maxIters);
	cudaErr = cudaMallocPitch((void**)&d_Paras, &parasPitch, sizeof(double)*paraSize, maxIters);
	cudaErr = cudaMemset2D(d_ColVec, colVecPitch, 0, sizeof(double)*minInliers, maxIters);
	cudaErr = cudaMemset2D(d_Paras, parasPitch, 0, sizeof(double)*paraSize, maxIters);
	GetHypoModelPara <<<grids, blocks>>> (d_hypoy, maxIters, minInliers, d_Amat, Aextent,
		colVecPitch, d_ColVec, parasPitch, d_Paras);
	//cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();
	cudaErr = cudaMemcpy2D(paraList, sizeof(double)*paraSize, d_Paras, parasPitch, sizeof(double)*paraSize, maxIters, cudaMemcpyDeviceToHost);
	cudaErr = cudaFree(d_ColVec);
	d_ColVec = NULL;
	free(h_hypoIDs);
	h_hypoIDs = NULL;
	free(h_hypox);
	h_hypox = NULL;
	free(h_hypoy);
	h_hypoy = NULL;
	cudaErr = cudaFree(d_hypox);
	d_hypox = NULL;
	cudaErr = cudaFree(d_hypoy);
	d_hypoy = NULL;

	/****************
	PART II:

	Modified inliers by the model which is defined by hypo-inliers, then estimated a new model by inlers.
	Furthermore, it should be checked whether the new model is enough better by the criteria: 
		1. enough inliers(such as inliers>0.9*pcsize)
		2. model error is small enough.
	By-product:
		1. Model error, 2. point to curve distances.

	*****************/
	double* d_xs, *d_ys;
	cudaErr = cudaMalloc((void**)&d_xs, sizeof(double)*pcSize);
	cudaErr = cudaMalloc((void**)&d_ys, sizeof(double)*pcSize);
	cudaErr = cudaMemcpy(d_xs, xvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_ys, yvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);

	/***********
	Step 2.1: Check out all inliers for the fitted model by using hypo-inliers in the proceding process.
	************/
	//Step 2.1.1: Prepared parameter spaces.
	int* d_inlierNum, *InlierNum;
	bool* d_bInOut, *bInOut;
	double* d_ModelErr, *d_Dists;
	size_t bInOutPitch, distPitch;
	InlierNum = (int*)malloc(sizeof(int)*maxIters);
	bInOut = (bool*)malloc(sizeof(bool)*pcSize*maxIters);
	cudaErr = cudaMalloc((void**)&d_inlierNum, sizeof(int)*maxIters);
	cudaErr = cudaMemset(d_inlierNum, 0, sizeof(int)*maxIters);
	cudaErr = cudaMalloc((void**)&d_ModelErr, sizeof(double)*maxIters);
	cudaErr = cudaMemset(d_ModelErr, 0, sizeof(double)*maxIters);
	cudaErr = cudaMallocPitch((void**)&d_bInOut,&bInOutPitch, sizeof(bool)*pcSize, maxIters);
	cudaErr = cudaMallocPitch((void**)&d_Dists, &distPitch, sizeof(double)*pcSize, maxIters);

	//Step 2.1.2: Get renewed inliers and estimate new model.
	CheckPointInOrOut <<<grids, blocks>>> (d_xs, d_ys, maxIters, pcSize,  paraSize, uTh, lTh,
		parasPitch, d_Paras, d_inlierNum, bInOutPitch, d_bInOut, d_ModelErr, distPitch, d_Dists);
	//cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();
	cudaErr = cudaMemcpy(InlierNum, d_inlierNum, sizeof(int)*maxIters, cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy2D(bInOut, sizeof(bool)*pcSize, d_bInOut, bInOutPitch, sizeof(bool)*pcSize, maxIters, cudaMemcpyDeviceToHost);

	//Step 2.1.3: Free spaces temporarily for reusing further.
	cudaErr = cudaFree(d_bInOut);
	d_bInOut = NULL;
	cudaErr = cudaFree(d_Amat.ptr);
	d_Amat.ptr = NULL;
	cudaErr = cudaFree(d_inlierNum);
	d_inlierNum = NULL;
	cudaErr = cudaFree(d_Paras);
	d_Paras = NULL;
	cudaErr = cudaFree(d_ModelErr);
	d_ModelErr = NULL;
	cudaErr = cudaFree(d_Dists);
	d_Dists = NULL;

	//Step 2.1.4 Resign new spaces:
	/**********
	IMPORTANT: In the part, there are TWO size will be modified,
	(1) The iteration number, from (maxIters) to (valuableIts);
	(2) The number of inlier points, for each iteration, this size will be various.
	***********/
	int valuableIts = 0;
	for (int i = 0; i < maxIters; i++)
	{
		if (InlierNum[i] > paraSize)
		{
			valuableIts++;
		}
	}
	
	int *ModifyInlierNum, maxInliers = 0;
	ModifyInlierNum = (int*)malloc(sizeof(int)*valuableIts);
	bool *ModifybInOut;
	ModifybInOut = (bool*)malloc(sizeof(bool*)*valuableIts*pcSize);
	
	int curIt = 0;
	for (int i = 0; i < maxIters; i++)
	{
		if (InlierNum[i] > paraSize)
		{
			ModifyInlierNum[curIt] = InlierNum[i];
			if (InlierNum[i] > maxInliers)
			{
				maxInliers = InlierNum[i];
			}
			cudaErr = cudaMemcpy(ModifybInOut+curIt*pcSize, bInOut+i*pcSize, sizeof(bool)*pcSize, cudaMemcpyHostToHost);
			curIt++;
		}
	}

	/***********
	Step 2.2: Setup matrices of As.
	************/
	cudaErr = cudaMalloc((void**)&d_inlierNum, sizeof(int)*valuableIts);
	cudaErr = cudaMemcpy(d_inlierNum, ModifyInlierNum, sizeof(int)*valuableIts, cudaMemcpyHostToDevice);
	cudaErr = cudaMallocPitch((void**)&d_bInOut, &bInOutPitch, sizeof(bool)*pcSize, valuableIts);
	cudaErr = cudaMemcpy2D(d_bInOut, bInOutPitch, ModifybInOut, sizeof(bool)*pcSize, sizeof(bool)*pcSize, valuableIts, cudaMemcpyHostToDevice);
	Aextent = make_cudaExtent(sizeof(double)*maxInliers, paraSize, valuableIts);
	cudaErr = cudaMalloc3D(&d_Amat, Aextent);
	SetupMatriceswithVariousSizes <<<grids, blocks>>> (d_xs, valuableIts, pcSize, paraSize, d_inlierNum,
		bInOutPitch, d_bInOut, Aextent, d_Amat);
	//cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();

	/***********
	Step 2.3: Get parameters results from As.
	************/
	double *d_CurYs, *d_Atvs;
	size_t curYsPitch, curAtvPitch;
	cudaErr = cudaMallocPitch((void**)&d_ColVec, &colVecPitch, sizeof(double)*maxInliers, valuableIts);
	cudaErr = cudaMallocPitch((void**)&d_Atvs, &curAtvPitch, sizeof(double)*paraSize, valuableIts);
	cudaErr = cudaMallocPitch((void**)&d_CurYs, &curYsPitch, sizeof(double*)*maxInliers, valuableIts);
	cudaErr = cudaMallocPitch((void**)&d_Paras, &parasPitch, sizeof(double)*paraSize, valuableIts);
	GetModelParawithVariateAs <<<grids, blocks>>> (d_ys, valuableIts, pcSize,  paraSize, Aextent, d_Amat,
		d_inlierNum, colVecPitch, d_ColVec, curAtvPitch, d_Atvs, curYsPitch, d_CurYs, bInOutPitch, d_bInOut, parasPitch, d_Paras);
	//cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();
	memset(paraList, 0, sizeof(double)*paraSize*maxIters);
	cudaErr = cudaMemcpy2D(paraList, sizeof(double)*paraSize, d_Paras, parasPitch, sizeof(double)*paraSize, valuableIts, cudaMemcpyDeviceToHost);
	cudaErr = cudaFree(d_ColVec);
	d_ColVec = NULL;
	cudaErr = cudaFree(d_CurYs);
	d_CurYs = NULL;
	cudaErr = cudaFree(d_Atvs);
	d_Atvs = NULL;

	/***********
	Step 2.4: Get model error and square distance.
	************/
	cudaErr = cudaMalloc((void**)&d_ModelErr, sizeof(double)*valuableIts);
	cudaErr = cudaMallocPitch((void**)&d_Dists, &distPitch, sizeof(double)*pcSize, valuableIts);
	GetModelSqDist <<<grids, blocks>>> (d_xs, d_ys, valuableIts, pcSize,  paraSize, parasPitch, d_Paras,
		d_inlierNum, d_ModelErr, distPitch, d_Dists);
	//cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();
	cudaErr = cudaMemcpy(modelErr, d_ModelErr, sizeof(double)*valuableIts, cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy2D(dists, sizeof(double)*pcSize, d_Dists, distPitch, sizeof(double)*pcSize, valuableIts, cudaMemcpyDeviceToHost);
	memset(resInliers, 0, sizeof(unsigned int)*maxIters);
	cudaErr = cudaMemcpy(resInliers, d_inlierNum, sizeof(int)*maxIters, cudaMemcpyDeviceToHost);
	resIters = valuableIts;

	/***********
	PART III: Free all RAM spaces both on CPU and GPU RAM as well. 
	************/
	//Host:
	free(InlierNum);
	InlierNum = NULL;
	free(bInOut);
	bInOut = NULL;
	free(ModifyInlierNum);
	ModifyInlierNum = NULL;
	free(ModifybInOut);
	ModifybInOut = NULL;
	
	//Device space:
	cudaErr = cudaFree(d_Amat.ptr);
	d_Amat.ptr = NULL;
	cudaErr = cudaFree(d_xs);
	d_xs = NULL;
	cudaErr = cudaFree(d_ys);
	d_ys = NULL;
	cudaErr = cudaFree(d_Paras);
	d_Paras = NULL;
	cudaErr = cudaFree(d_bInOut);
	d_bInOut = NULL;
	cudaErr = cudaFree(d_ModelErr);
	d_ModelErr = NULL;
	cudaErr = cudaFree(d_Dists);
	d_Dists = NULL;
	cudaErr = cudaFree(d_inlierNum);
	d_inlierNum = NULL;
	return cudaErr;
}

__global__ void ConvertF2IMat(float* fVals, int fLen, size_t uiPitch, size_t width, size_t uiMax, size_t *uiVals)
{
	int bID = blockIdx.x + blockIdx.y*gridDim.x;
	int tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tID < fLen)
	{
		size_t* pLine = (size_t*)((char*)uiVals + (tID / width)*uiPitch);
		if (fVals[tID] < 1e-6)
		{
			uiVals[tID] = 0;
		}
		else
		{
			pLine[tID%width] = size_t((fVals[tID] - 1e-6) / (1 - 1e-6)*uiMax);
		}
	}
}

__global__ void GetHypoXYAtOnce(double* xs, double* ys, size_t* hypoID2D, size_t idPitch, size_t width, size_t height, double* hypoxs, size_t xPitch, double* hypoys, size_t yPitch)
{
	unsigned int bID = blockIdx.y*gridDim.x + blockIdx.x,
		tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	if (tID < width*height)
	{
		unsigned int curRow, curCol;
		curRow = tID / width;
		curCol = tID%width;
		hypoxs[curRow*xPitch / sizeof(double) + curCol] =
			xs[hypoID2D[curRow*idPitch / sizeof(size_t) + curCol]];
		hypoys[curRow*yPitch / sizeof(double) + curCol] =
			ys[hypoID2D[curRow*idPitch / sizeof(size_t) + curCol]];
	}
}

__device__ void Merge(double* xs, double* ys, double* bx, double* by, int low, int mid, int high)
{
	int lb = mid - low, lc = high - mid;
	if (!bx || !by)
	{
		return;
	}

	double* ax = xs + low, *ay = ys + low, *cx = xs + mid, *cy = ys + mid;

	for (int i = 0; i < lb; i++)
	{
		bx[i] = ax[i];
		by[i] = ay[i];
	}

	for (int i = 0, j = 0, k = 0; j < lb || k < lc;)
	{
		if (j < lb && (lc <= k || bx[j] < cx[k]))
		{
			ax[i] = bx[j];
			ay[i] = by[j];
			i++;
			j++;
		}

		if (k < lc && (lb <= j || cx[k] <= bx[j]))
		{
			ax[i] = cx[k];
			ay[i] = cy[k];
			i++;
			k++;
		}
	}
}

__device__ void MergePass(double* xs, double* ys, double* tmpx, double* tmpy, int k, int len)
{
	int i = 0;
	while (i + 2 * k <= len)
	{
		Merge(xs, ys, tmpx, tmpy, i, i + k, i + 2 * k);
		i += 2 * k;
	}
	if (i + k <= len)
		Merge(xs, ys, tmpx, tmpy, i, i + k, len);
}

__device__ void MergeSortAlong(double* xs, double* ys, char axisName, double* tmpx, double* tmpy, int len)
{
	
	double* tgX, *tgY;
	if (axisName == 'x' || axisName == 'X')
	{
		tgX = xs;
		tgY = ys;
	}
	else
	{
		tgX = ys;
		tgY = xs;
	}

	int k = 1;
	while (k < len)
	{
		MergePass(tgX, tgY, tmpx, tmpy, k, len);
		k *= 2;
	}
}

__device__ void QSortAlong(double* xs, double* ys, char axisName, int left, int right)
{
	double* tgX, *tgY;
	if (axisName == 'x' || axisName == 'X')
	{
		tgX = xs;
		tgY = ys;
	}
	else
	{
		tgX = ys;
		tgY = xs;
	}

	if (left >= right)
		return;

	int i = left, j = right;
	double keyX = tgX[i], keyY = tgY[i];
	while (i < j)
	{
		while (i < j && keyX <= tgX[j])
		{
			j--;
		}
		tgX[i] = tgX[j];
		tgY[i] = tgY[j];

		while (i < j && keyX >= tgX[i])
		{
			i++;
		}
		tgX[j] = tgX[i];
		tgY[j] = tgY[i];
	}
	tgX[i] = keyX;
	tgY[i] = keyY;
	QSortAlong(tgX, tgY, 'x', left, i - 1);
	QSortAlong(tgX, tgY, 'x', i + 1, right);
}

__global__ void SortHypoXY(CtrPtBound inBd, size_t width, size_t height, double* hypoxs, size_t xPitch, double* hypoys, size_t yPitch)
{
	unsigned int bID = blockIdx.y*gridDim.x + blockIdx.x,
		tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	if (tID < height)
	{
		double* xs = (double*)((char*)hypoxs + xPitch*tID), *ys = (double*)((char*)hypoys + yPitch*tID);
		//QSortAlong(xs, ys, 'x', 0, width - 1);

		double* tmpx = (double*)malloc(width * sizeof(double)), *tmpy = (double*)malloc(width * sizeof(double));
		MergeSortAlong(xs, ys, 'x', tmpx, tmpy, width);
		xs[0] = inBd.xbeg;
		xs[width - 1] = inBd.xend;
		ys[0] = inBd.ybeg;
		ys[width - 1] = inBd.yend;
		free(tmpx);
		tmpx = NULL;
		free(tmpy);
		tmpy = NULL;
	}
}

__global__ void GetHypoXYs(double* xs, double* ys, size_t* hypoID2D, size_t idPitch, size_t width, size_t height, double* hypoxs, size_t xPitch, double* hypoys, size_t yPitch)
{
	unsigned int tColID = blockIdx.x*blockDim.x + threadIdx.x,
		tRowID = blockIdx.y*blockDim.y + threadIdx.y,
		gThWidth = gridDim.x*blockDim.x, gThHeight = gridDim.y*blockDim.y;
	unsigned int bColLimit = ((width + blockDim.x - 1) / blockDim.x < gridDim.x) ? (width + blockDim.x - 1) / blockDim.x : gridDim.x,
		bRowLimit = ((height + blockDim.y - 1) / blockDim.y < gridDim.y) ? (height + blockDim.y - 1) / blockDim.y : gridDim.y;

	if (blockIdx.x < bColLimit && blockIdx.y < bRowLimit)
	{
		unsigned int tColMove = (width + gThWidth - 1) / gThWidth, tRowMove = (height + gThHeight - 1) / gThHeight,
			curCol = 0, curRow = 0;
		for (unsigned int colMvI = 0; colMvI < tColMove; colMvI++)
		{
			for (unsigned int rowMvI = 0; rowMvI < tRowMove; rowMvI++)
			{
				curCol = colMvI*gThWidth + tColID;
				curRow = rowMvI*gThHeight + tRowID;
				if (curCol < width && curRow < height)
				{
					hypoxs[curRow*xPitch / sizeof(double) + curCol] =
						xs[hypoID2D[curRow*idPitch / sizeof(size_t) + curCol]];
					hypoys[curRow*yPitch / sizeof(double) + curCol] =
						ys[hypoID2D[curRow*idPitch / sizeof(size_t) + curCol]];
				}
			}
		}
	}
}

__global__ void SetupSingleMatrix(double* xs, size_t* hypoID2D, size_t linePitch, size_t width,size_t height, size_t lineID, int paraSize,size_t APitch, double* Amat)
{
	int bID = blockIdx.x + blockIdx.y*gridDim.x;
	int tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	size_t* curHypoLine = NULL;
	if (lineID < height)
	{
		curHypoLine = (size_t*)((char*)hypoID2D + lineID*linePitch);
	}
	if (NULL != curHypoLine)
	{
		extern __shared__ double curX[];
		int tIDInB = threadIdx.y*blockDim.x + threadIdx.x;
		unsigned int bSize = blockDim.x*blockDim.y;
		if (bSize >= width)
		{
			if (tIDInB < width)
			{
				curX[tIDInB] = xs[curHypoLine[tIDInB]];
			}
		}
		else
		{
			for (unsigned int ii = 0; ii < bSize / width; ii++)
			{
				curX[ii*bSize + tIDInB] = xs[curHypoLine[ii*bSize + tIDInB]];
			}
			if (tIDInB < bSize%width)
			{
				curX[bSize / width*bSize + tIDInB] = xs[curHypoLine[bSize / width*bSize + tIDInB]];
			}
		}
		__syncthreads();
		if (tID < width*paraSize)
		{
			unsigned int row = 0, col = 0;
			row = (unsigned int)(tID / paraSize);
			col = (unsigned int)tID%paraSize;
			double tmpVal = 1.0;
			for (unsigned int ii = 1; ii <= col; ii++)
			{
				tmpVal *= curX[row];
			}
			double* curALine = (double*)((char*)Amat + row*APitch);
			curALine[col] = tmpVal;
		}
	}
}

__global__ void SetIdentityMat(double* inoutMat, size_t matPitch, size_t n)
{
	int bID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < n)
	{
		double* ioMatRow = (double*)((char*)inoutMat + tid*matPitch);
		ioMatRow[tid] = 1.0;
	}
}

__global__ void GetHk(size_t colID, double* Amat, size_t APitch, size_t widthA, size_t heightA, double* Hmat, size_t HPitch)
{
	if (blockIdx.x < (heightA + blockDim.x - 1) / blockDim.x && blockIdx.y < (heightA + blockDim.y - 1) / blockDim.y)
	{
		extern __shared__ double inShared[];
		double* colVec = (double*)inShared, *pBeta = (double*)&colVec[heightA];
		unsigned int tIDInB = threadIdx.y*blockDim.x + threadIdx.x;
		unsigned int bSize = blockDim.x*blockDim.y;

		double* curALine;
		if (bSize >= heightA)
		{
			if (tIDInB < heightA)
			{
				curALine = (double*)((char*)Amat + tIDInB*APitch);
				colVec[tIDInB] = curALine[colID];
			}
		}
		else
		{
			for (unsigned int ii = 0; ii < heightA / bSize; ii++)
			{
				curALine = (double*)((char*)Amat + (ii*bSize + tIDInB)*APitch);
				colVec[ii*bSize + tIDInB] = curALine[colID];
			}
			if (tIDInB < heightA%bSize)
			{
				curALine = (double*)((char*)Amat + (bSize* (heightA / bSize) + tIDInB)*APitch);
				colVec[bSize* (heightA / bSize) + tIDInB] = curALine[colID];
			}
		}

		__syncthreads();

		double sigma = 0.0, norm = 0.0;
		if (tIDInB == 0)
		{
			for (size_t ii = colID; ii < heightA; ii++)
			{
				norm += colVec[ii] * colVec[ii];
			}
			norm = sqrt(norm);
			if (colVec[colID] >= 0)
			{
				sigma = norm;
			}
			else
			{
				sigma = -norm;
			}

			*pBeta = 1 / sigma / (sigma + colVec[colID]);
			colVec[colID] += sigma;
		}

		__syncthreads();

		unsigned int row = threadIdx.y + blockIdx.y*blockDim.y, col = threadIdx.x + blockIdx.x*blockDim.x;
		if (row < heightA && col < heightA)
		{
			double tmpH = 0.0;
			if (row < colID || col < colID)
			{
				if (row == col)
				{
					tmpH = 1.0;
				}
				else
				{
					tmpH = 0.0;
				}
			}
			else
			{
				if (row == col)
				{
					tmpH = 1.0 - (*pBeta)*colVec[row] * colVec[col];
				}
				else
				{
					tmpH = -(*pBeta)*colVec[row] * colVec[col];
				}
			}
			double* curHLine = (double*)((char*)Hmat + row*HPitch);
			curHLine[col] = tmpH;
		}
	}
}

__global__ void MatMultMat(double* LMat, size_t LPitch, size_t LRows, size_t LCols, double* RMat, size_t RPitch, size_t RRows, size_t RCols,
	double* ResMat, size_t ResPitch)
{
	extern __shared__ double totalArr[];
	double* LTile = totalArr, *RTile = (double*)&LTile[blockDim.y*LCols], *inTile;//Row-major storage
	unsigned int tRowID = blockIdx.y*blockDim.y + threadIdx.y, tColID = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int  LTileRow = 0, LTileCol = 0, RTileRow = 0, RTileCol = 0;
	unsigned int sTileID = 0, gTileID = 0;

	if (blockIdx.y <= (unsigned int)LRows / blockDim.y && blockIdx.x <= (unsigned int)RCols / blockDim.x)
	{
		if ((blockIdx.y + 1) * blockDim.y <= (unsigned int)LRows)
		{
			LTileRow = blockDim.y;
		}
		else
		{
			LTileRow = (unsigned int)LRows%blockDim.y;
		}
		LTileCol = (unsigned int)LCols;
		inTile = (double*)((char*)LMat + blockIdx.y*blockDim.y*LPitch);
		unsigned int ii = 0, realCol = 0;
		for (ii = 0; ii <= LTileCol / blockDim.x; ii++)
		{
			if ((ii + 1)*blockDim.x <= (unsigned int)LTileCol)
			{
				realCol = blockDim.x;
			}
			else
			{
				realCol = (unsigned int)LTileCol%blockDim.x;
			}
			if (threadIdx.y < LTileRow && threadIdx.x < realCol)
			{
				sTileID = threadIdx.y*LTileCol + ii*blockDim.x + threadIdx.x;
				gTileID = threadIdx.y*LPitch / sizeof(double) + ii*blockDim.x + threadIdx.x;
				LTile[sTileID] = inTile[gTileID];
			}
		}

		if ((blockIdx.x + 1) * blockDim.x <= (unsigned int)RCols)
		{
			RTileCol = blockDim.x;
		}
		else
		{
			RTileCol = (unsigned int)RCols%blockDim.x;
		}
		RTileRow = (unsigned int)RRows;
		inTile = RMat;
		unsigned int realRow = 0;
		for (ii = 0; ii <= RTileRow / blockDim.y; ii++)
		{
			if ((ii + 1)*blockDim.y <= RTileRow)
			{
				realRow = blockDim.y;
			}
			else
			{
				realRow = (unsigned int)RTileRow%blockDim.y;
			}
			if (threadIdx.x < RTileCol && threadIdx.y < realRow)
			{
				sTileID = ii*RTileCol*blockDim.y + threadIdx.y*RTileCol + threadIdx.x;
				gTileID = (ii*blockDim.y + threadIdx.y)*RPitch / sizeof(double) + blockIdx.x*blockDim.x + threadIdx.x;
				RTile[sTileID] = inTile[gTileID];
			}
		}
		__syncthreads();

		if (tRowID<LRows && tColID<RCols)
		{
			double sum = 0.0;

			for (unsigned int ii = 0; ii < LCols; ii++)
			{
				//Modified for Rows and columns.
				sum += LTile[threadIdx.y*LCols + ii] * RTile[ii*RTileCol + threadIdx.x];
			}

			double *resLine = (double*)((char*)ResMat + tRowID*ResPitch);
			/*if (CHECK_D_ZERO(sum))
			{
				resLine[tColID] = 0.0;
			}
			else
			{*/
				resLine[tColID] = sum;
			//}	
		}
	}
}

__global__ void MatTranspose(double* inMat, size_t inPitch, size_t inRows, size_t inCols, double* outMat, size_t outPitch)
{
	unsigned int tColID = blockIdx.x*blockDim.x + threadIdx.x,
		tRowID = blockIdx.y*blockDim.y + threadIdx.y,
		gThWidth = gridDim.x*blockDim.x, gThHeight = gridDim.y*blockDim.y;
	unsigned int blockValColLimit = ((inCols + blockDim.x - 1) / blockDim.x < gridDim.x) ? (inCols + blockDim.x - 1) / blockDim.x : gridDim.x,
		blockValRowLimit = ((inRows + blockDim.y - 1) / blockDim.y < gridDim.y) ? (inRows + blockDim.y - 1) / blockDim.y : gridDim.y;

	if (blockIdx.x < blockValColLimit && blockIdx.y < blockValRowLimit)
	{
		unsigned int tColMove = (inCols + gThWidth - 1) / gThWidth, tRowMove = (inRows + gThHeight - 1) / gThHeight,
			curCol = 0, curRow = 0;
		for (unsigned int colMvI = 0; colMvI < tColMove; colMvI++)
		{
			for (unsigned int rowMvI = 0; rowMvI < tRowMove; rowMvI++)
			{
				curCol = colMvI*gThWidth + tColID;
				curRow = rowMvI*gThHeight + tRowID;
				if (curCol < inCols && curRow < inRows)
				{
					outMat[curCol*outPitch / sizeof(double) + curRow] =
						inMat[curRow*inPitch / sizeof(double) + curCol];
				}
			}
		}
	}
}

__global__ void MatMultVec(double* inMat, size_t matPitch, size_t matRows, size_t matCols, double* inVec, double* outVec)
{
	extern __shared__ double totalArr[];
	double* matTile = totalArr, *vecTile = (double*)&matTile[matCols*blockDim.x*blockDim.y];

	unsigned int needBlocks = (matRows + blockDim.x*blockDim.y - 1) / (blockDim.x*blockDim.y),
		bID = blockIdx.x + blockIdx.y*gridDim.x;

	if (bID < needBlocks)
	{
		unsigned int tIDInB = threadIdx.x + threadIdx.y*blockDim.x, tidLimit = blockDim.x*blockDim.y;
		if ((bID + 1)*blockDim.x*blockDim.y > matRows)
		{
			tidLimit = matRows % (blockDim.x*blockDim.y);
		}

		if (tIDInB < tidLimit)
		{
			double* matTileLine = (double*)((char*)inMat + tIDInB*matPitch);
			for (unsigned int ii = 0; ii < matCols; ii++)
			{
				matTile[tIDInB*matCols + ii] = matTileLine[ii];
			}
		}

		if (tIDInB < matCols)
		{
			vecTile[tIDInB] = inVec[tIDInB];
		}

		__syncthreads();

		if (tIDInB < tidLimit)
		{
			double tmpVal = 0.0;
			for (unsigned int ii = 0; ii < matCols; ii++)
			{
				tmpVal += matTile[tIDInB*matCols + ii] * vecTile[ii];
			}
			outVec[bID*blockDim.x*blockDim.y + tIDInB] = tmpVal;
		}
	}
}


__global__ void SolveRMatSystem(double* Rmat, size_t RPitch, size_t valLen, double* inYs, double* outRes)
{
	unsigned int bID = blockIdx.y*gridDim.x + blockIdx.x,
		tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;

	if (tID == 0)
	{
		double tmpVal = 0.0;
		for (int ii = valLen - 1; ii >= 0; ii--)
		{
			tmpVal = 0.0;
			for (int jj = ii + 1; jj < valLen; jj++)
			{
				tmpVal += Rmat[ii*RPitch / sizeof(double) + jj] * outRes[jj];
			}
			outRes[ii] = (inYs[ii] - tmpVal) / Rmat[ii*RPitch / sizeof(double) + ii];
		}
	}
}


__global__ void ComputeInlierNum(double* xvals, double* yvals,size_t pcsize, double* paras,size_t parasize,
	double UTh, double LTh, bool* bInOut)
{
	unsigned int bSize = blockDim.x*blockDim.y,
		bID = blockIdx.x + blockIdx.y*gridDim.x,
		tIDInB = threadIdx.x + threadIdx.y*blockDim.x,
		tID = bID*bSize + tIDInB;
	extern __shared__ double totalVal[];
	double* xTile = totalVal, *yTile = &xTile[bSize], *xpow = &yTile[bSize], *paraTile=&xpow[bSize];
	unsigned int *inNum = (unsigned int*)&paraTile[parasize];
	unsigned int needBlocks = (unsigned int)(pcsize + bSize - 1) / bSize;
	if (bID < needBlocks)
	{
		double* xGlobal = xvals + bID*bSize, *yGlobal = yvals + bID*bSize;
		unsigned int realLen = bSize;
		if (bID == needBlocks - 1)
		{
			realLen = (unsigned int)pcsize%bSize;
		}

		if (tIDInB < parasize)
		{
			paraTile[tIDInB] = paras[tIDInB];
		}
		
		if (tIDInB == 0)
		{
			inNum[0] = (unsigned int)0;
		}
		
		__syncthreads();

		if (tIDInB < realLen)
		{
			xTile[tIDInB] = xGlobal[tIDInB];
			yTile[tIDInB] = yGlobal[tIDInB];
			xpow[tIDInB] = 1.0;
			for (size_t ii = 0; ii < parasize; ii++)
			{
				yTile[tIDInB] -= xpow[tIDInB] * paraTile[ii];
				xpow[tIDInB] *= xTile[tIDInB];
			}
			if ((yTile[tIDInB] > 0 && yTile[tIDInB] - UTh < 0.0) || (yTile[tIDInB]<0 && yTile[tIDInB] + LTh>0.0))
			{
				bInOut[tID] = true;
			}
			else
			{
				bInOut[tID] = false;
			}
		}
	}
}

__global__ void SeriesSum(unsigned int* numSeries, size_t len, size_t curIt, unsigned int *res)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y,
		tIDInB = threadIdx.x + threadIdx.y*blockDim.x,
		bSize = blockDim.x*blockDim.y;
	extern __shared__ unsigned int inSeries[];
	__shared__ unsigned int tmpVal[1];
	unsigned int realLen = bSize;
	if (bID == 0)
	{
		for (unsigned int ii = 0; ii <= (unsigned int)len / bSize; ii++)
		{
			if (ii == (unsigned int)len / bSize)
			{
				realLen = (unsigned int)len%bSize;
			}
			if (tIDInB < realLen)
			{
				inSeries[tIDInB] = numSeries[ii*bSize + tIDInB];
			}
			__syncthreads();
			if (tID == 0)
			{
				for (size_t ii = 0; ii < realLen; ii++)
				{
					tmpVal[0] += inSeries[ii];
				}
				res[curIt] = tmpVal[0];
			}
		}
	}
}

__global__ void GetBestReslut(unsigned int* inLiers, size_t inlierLen, double* curParas, size_t paraSize, size_t pcSize, bool* bInOut, double* bestParas, double* bestRate, int* inlierIDs)
{
	unsigned int bSize = blockDim.x*blockDim.y,
		bID = blockIdx.x + blockIdx.y*gridDim.x,
		tIDInB = threadIdx.x + threadIdx.y*blockDim.x,
		tID = bID*bSize + tIDInB;
	__shared__ size_t curinLiers[1];
	if (tIDInB == 0)
	{
		curinLiers[0] = 0;
		for (size_t ii = 0; ii < inlierLen; ii++)
		{
			curinLiers[0] += inLiers[ii];
		}
	}
	__syncthreads();
	double curRate = curinLiers[0] * 1.0 / (pcSize*1.0);
	if (curRate > bestRate[0])
	{
		if (bID == 0)
		{
			if (tIDInB == 0)
			{
				bestRate[0] = curRate;
			}
			
			if(tIDInB<(unsigned int)paraSize)
			{
				bestParas[tIDInB] = curParas[tIDInB];
			}
		}

		if (tID <(unsigned int)pcSize )
		{
			if (bInOut[tID])
			{
				inlierIDs[tID] = tID;
			}
			else
			{
				inlierIDs[tID] = -1;
			}
		}
	}
	__syncthreads();
}

__global__ void GetBestResult(unsigned int* tmpLen, double* curParas, size_t paraSize, size_t pcSize, bool* bInOut, double* bestParas, double* bestRate, int* inlierIDs)
{
	unsigned int bSize = blockDim.x*blockDim.y,
		bID = blockIdx.x + blockIdx.y*gridDim.x,
		tIDInB = threadIdx.x + threadIdx.y*blockDim.x,
		tID = bID*bSize + tIDInB;
	size_t lenLimit = pcSize, span = pcSize / 2;
	bool bFirst = true;
	while (lenLimit >= 2)
	{
		if (tID < span)
		{
			if (bFirst)
			{
				if (bInOut[tID])
				{
					tmpLen[tID] = 1;
					inlierIDs[tID] = tID;
				}
				else
				{
					tmpLen[tID] = 0;
					inlierIDs[tID] = -1;
				}
				if (bInOut[tID + span])
				{
					tmpLen[tID]++;
					inlierIDs[tID + span] = tID + span;
				}
				else
				{
					inlierIDs[tID + span] = -1;
				}
				bFirst = false;
			}
			else
			{
				tmpLen[tID] += tmpLen[tID + span];
			}

			if (lenLimit % 2 != 0)
			{
				if (tID == span - 1)
				{
					tmpLen[tID] += tmpLen[tID + span + 1];
				}
			}
		}
		__syncthreads();
		lenLimit /= 2;
		span = lenLimit / 2;
	}

	double curRate;
	if (bID == 0)
	{
		if (tIDInB == 0)
		{
			curRate = tmpLen[0] * 1.0 / (pcSize*1.0);
			if (curRate > bestRate[0])
			{
				bestRate[0] = curRate;
			}
		}

		if (tIDInB < (unsigned int)paraSize)
		{
			bestParas[tID] = curParas[tID];
		}
	}
}

template<class T>
void GetGridFactor(T totalNum,T blockSize, T& outNum1, T& outNum2)
{
	T inNum = T((totalNum + blockSize - 1) / blockSize);
	if (inNum > T(0))
	{
		unsigned int maxIt = floor(sqrt(inNum));
		outNum1 = T(1);
		outNum2 = T(1);
		for (unsigned int ii = maxIt; ii >= 2; ii--)
		{
			if (inNum%ii == 0)
			{
				outNum1 = T(ii);
				break;
			}
		}
		if (outNum1 > 1)
		{
			outNum2 = inNum / outNum1;
		}
		else
		{
			outNum1 = inNum;
		}
		
	}
}

extern "C" cudaError_t RANSACOnGPU1(__in double* xvals, __in double* yvals, __in size_t pcSize, __in int maxIters, __in int minInliers, __in int paraSize, __in double uTh, __in double lTh,
	__out double* &bestParas, __out int* &resInliers, __out double &modelErr, __out double* &dists)
{
	cudaError_t cudaErr;
	unsigned int gridX = 1, gridY = 1;
	GetGridFactor((unsigned int)maxIters, (unsigned int)THREAD_SIZE, gridX, gridY);
	dim3 blocks(THREAD_SIZE, THREAD_SIZE), grids(gridX, gridY);
	//Copy total data from host into device:
	double* d_xs, *d_ys;
	cudaErr = cudaMalloc((void**)&d_xs, sizeof(double)*pcSize);
	cudaErr = cudaMalloc((void**)&d_ys, sizeof(double)*pcSize);
	cudaErr = cudaMemcpy(d_xs, xvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_ys, yvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);

	int curIt = 0, bestLast = 0;
	const int bestExit = maxIters*0.1;

	double bestRate = 0.0;
	float *d_Randf, *h_Randf;
	cudaErr = cudaMalloc((void**)&d_Randf, minInliers * maxIters * sizeof(float));
	h_Randf = new float[minInliers*maxIters];
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, unsigned long long(time(0)));
	curandGenerateUniform(gen, d_Randf, minInliers * maxIters);
 	cudaErr = UserDebugCopy(h_Randf, d_Randf, minInliers * maxIters * sizeof(float), cudaMemcpyDeviceToHost);
 

	size_t *d_HypoIDs, *h_HypoIDs;
	size_t hypoPitch;
	cudaErr = cudaMallocPitch((void**)&d_HypoIDs, &hypoPitch, minInliers * sizeof(size_t), maxIters);
	h_HypoIDs = new size_t[minInliers*maxIters];
	blocks = dim3(32, 32);
	GetGridFactor((unsigned int)(maxIters*minInliers),(unsigned int)32*32, grids.x, grids.y);
	ConvertF2IMat<<<grids,blocks>>>(d_Randf, minInliers*maxIters, hypoPitch, minInliers, pcSize, d_HypoIDs);
	cudaErr = UserDebugCopy2D(h_HypoIDs, minInliers * sizeof(size_t), d_HypoIDs, hypoPitch, minInliers * sizeof(size_t), maxIters, cudaMemcpyDeviceToHost);

	double *d_HypoXs, *d_HypoYs, *h_HypoXs, *h_HypoYs, *d_curHypoys, *h_curHypoys;
	size_t hypoXPitch, hypoYPitch;
	cudaErr = cudaMallocPitch((void**)&d_HypoXs, &hypoXPitch, minInliers * sizeof(double), maxIters);
	cudaErr = cudaMallocPitch((void**)&d_HypoYs, &hypoYPitch, minInliers * sizeof(double), maxIters);
	cudaErr = cudaMalloc((void**)&d_curHypoys, minInliers * sizeof(double));
	h_HypoXs = new double[minInliers*maxIters];
	h_HypoYs = new double[minInliers*maxIters];
	h_curHypoys = new double[minInliers];
	//grids.x = 10, grids.y = 10;
	//GetHypoXYs<<<grids,dim3(28,32)>>>(d_xs, d_ys, d_HypoIDs, hypoPitch, minInliers, maxIters, d_HypoXs, hypoXPitch, d_HypoYs, hypoYPitch);
	//cudaErr = UserDebugCopy2D(h_HypoXs, minInliers * sizeof(double), d_HypoXs, hypoXPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);
	//cudaErr = UserDebugCopy2D(h_HypoYs, minInliers * sizeof(double), d_HypoYs, hypoYPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);
	blocks = dim3(32, 32);
	GetGridFactor((unsigned int)minInliers*maxIters, (unsigned int)32 * 32, grids.x, grids.y);
	GetHypoXYAtOnce << <grids, blocks >> > (d_xs, d_ys, d_HypoIDs, hypoPitch, minInliers, maxIters, d_HypoXs, hypoXPitch, d_HypoYs, hypoYPitch);
	cudaErr = UserDebugCopy2D(h_HypoXs, minInliers * sizeof(double), d_HypoXs, hypoXPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);
	cudaErr = UserDebugCopy2D(h_HypoYs, minInliers * sizeof(double), d_HypoYs, hypoYPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);

	double* d_Amat, *h_Amat;
	size_t APitch;
	cudaErr = cudaMallocPitch((void**)&d_Amat, &APitch, paraSize * sizeof(double), minInliers);
	h_Amat = new double[minInliers*paraSize];

	double* d_Hmat, *h_Hmat;
	size_t HPitch;
	cudaErr = cudaMallocPitch((void**)&d_Hmat, &HPitch, minInliers * sizeof(double), minInliers);
	h_Hmat = new double[minInliers*minInliers];

	double* d_curParas, *h_curParas,*d_bestParas, *h_bestParas;
	cudaErr = cudaMalloc((void**)&d_curParas, paraSize * sizeof(double));
	cudaErr = cudaMalloc((void**)&d_bestParas, paraSize * sizeof(double));
	h_curParas = new double[paraSize];
	h_bestParas = new double[paraSize];

	dim3 inlierBlock(8, 16), inlierGrids(1, 1, 1);
	inlierGrids.x = (pcSize + inlierBlock.x*inlierBlock.y - 1) / (inlierBlock.x*inlierBlock.y);
	int *d_inlierIDs,*h_inlierIDs;
	cudaErr = cudaMalloc((void**)&d_inlierIDs, pcSize * sizeof(size_t));
	h_inlierIDs = new int[pcSize];

	bool* d_bInOut, *h_bInOut;
	cudaErr = cudaMalloc((void**)&d_bInOut, pcSize * sizeof(bool));
	h_bInOut = new bool[pcSize];
	double* d_bestRate;
	cudaErr = cudaMalloc((void**)&d_bestRate, sizeof(double));

	unsigned int *d_tmpLen;
	cudaErr = cudaMalloc((void**)&d_tmpLen, pcSize / 2 * sizeof(unsigned int));
	
	while (curIt < maxIters && bestLast < bestExit)
	{
		//Generate A and y by hypo-inliers
		cudaErr = cudaMemset2D(d_Amat, APitch, 0.0, paraSize * sizeof(double), minInliers);
		blocks = dim3(THREAD_SIZE, THREAD_SIZE);
		GetGridFactor((unsigned int)minInliers*paraSize, blocks.x*blocks.y, grids.x, grids.y);
		SetupSingleMatrix<<<grids, blocks, minInliers*sizeof(double)>>>(d_xs, d_HypoIDs, hypoPitch, minInliers, maxIters, curIt, paraSize, APitch, d_Amat);
		cudaErr = UserDebugCopy2D(h_Amat, paraSize * sizeof(double), d_Amat, APitch, paraSize * sizeof(double), minInliers, cudaMemcpyDeviceToHost);
		cudaErr = cudaMemcpy(d_curHypoys, d_HypoYs+curIt*hypoYPitch/sizeof(double), minInliers * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaErr = UserDebugCopy(h_curHypoys, d_curHypoys, minInliers * sizeof(double), cudaMemcpyDeviceToHost);

		//Solve A*alpha=y on hypo-inliers 
		GetGridFactor((unsigned int)minInliers, blocks.x*blocks.y, grids.x, grids.y);
		cudaErr = cudaMemset2D(d_Hmat, HPitch, 0.0, minInliers * sizeof(double), minInliers);
		SetIdentityMat <<<grids, blocks >>> (d_Hmat, HPitch, minInliers);
		cudaErr = UserDebugCopy2D(h_Hmat, minInliers * sizeof(double), d_Hmat, HPitch, minInliers * sizeof(double), minInliers, cudaMemcpyDeviceToHost);
		
		//A(k+1)=H(k)*A(k), Q(k+1)=H(k+1)*Q(k)
		//Finally,it obtains R=A(n) and Q=Q(n),while the linear system A*alpha=y is converted to R*alpha=Q^t*y.
		blocks = dim3(8, 8, 1);
		for (size_t ii = 0; ii < paraSize; ii++)
		{
			grids.x = (minInliers + blocks.x - 1) / blocks.x;
			grids.y = (minInliers + blocks.y - 1) / blocks.y;
			GetHk<<<grids, blocks, (minInliers + 1) * sizeof(double)>>>(ii, d_Amat, APitch, paraSize, minInliers, d_Hmat, HPitch);
			cudaErr = UserDebugCopy2D(h_Hmat, minInliers * sizeof(double), d_Hmat, HPitch, minInliers * sizeof(double), minInliers, cudaMemcpyDeviceToHost);
			grids.x = (paraSize + blocks.x - 1) / blocks.x;
			//grids.y = (minInliers + blocks.y - 1) / blocks.y;
			MatMultMat<<<grids, blocks, minInliers*(blocks.x + blocks.y) * sizeof(double)>>> (d_Hmat, HPitch, minInliers, minInliers, d_Amat, APitch, minInliers, paraSize, d_Amat, APitch);
			cudaErr = UserDebugCopy2D(h_Amat, paraSize * sizeof(double), d_Amat, APitch, paraSize * sizeof(double), minInliers, cudaMemcpyDeviceToHost);

			blocks.x = 4, blocks.y = 4;
			grids.x = 1;
			grids.y = (minInliers + blocks.x*blocks.y - 1) / (blocks.x*blocks.y);
			MatMultVec <<<grids, blocks, minInliers*(blocks.x*blocks.y + 1) * sizeof(double) >>>(d_Hmat, HPitch, minInliers, minInliers, d_curHypoys, d_curHypoys);
			cudaErr = UserDebugCopy(h_curHypoys, d_curHypoys, minInliers * sizeof(double), cudaMemcpyDeviceToHost);
		}
		SolveRMatSystem<<<1,1>>>(d_Amat, APitch, paraSize, d_curHypoys, d_curParas);
		cudaErr = UserDebugCopy(h_curParas, d_curParas, paraSize * sizeof(double), cudaMemcpyDeviceToHost);

		//Check inliers on total data
		ComputeInlierNum<<<inlierGrids,inlierBlock,(3*inlierBlock.x*inlierBlock.y+paraSize)*sizeof(double)+sizeof(unsigned int)>>>(d_xs, d_ys, pcSize, d_curParas, paraSize, uTh, lTh, d_bInOut);
		cudaErr = UserDebugCopy(h_bInOut, d_bInOut, pcSize * sizeof(bool), cudaMemcpyDeviceToHost);

		blocks = dim3(16, 16);
		grids = dim3((pcSize + 255) / 256, 1, 1);
		GetBestResult<<<grids,blocks>>>(d_tmpLen,d_curParas, paraSize, pcSize, d_bInOut, d_bestParas, d_bestRate, d_inlierIDs);
		cudaErr = UserDebugCopy(&bestRate, d_bestRate, sizeof(double), cudaMemcpyDeviceToHost);
		cudaErr = UserDebugCopy(h_curParas, d_bestParas, paraSize * sizeof(double), cudaMemcpyDeviceToHost);
		cudaErr = UserDebugCopy(h_inlierIDs, d_inlierIDs, pcSize * sizeof(int), cudaMemcpyDeviceToHost);
		curIt++;
	}
	cudaErr = cudaMemcpy(bestParas, d_bestParas, paraSize * sizeof(double), cudaMemcpyDeviceToHost);

	cudaErr = cudaFree(d_tmpLen);
	d_tmpLen = NULL;

	cudaErr = cudaMemcpy(resInliers, d_inlierIDs, pcSize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaErr = cudaFree(d_bestRate);
	d_bestRate = NULL;

	cudaErr = cudaFree(d_bInOut);
	d_bInOut = NULL;
	delete[] h_bInOut;
	h_bInOut = NULL;

	cudaErr = cudaFree(d_inlierIDs);
	d_inlierIDs = NULL;
	delete[] h_inlierIDs;
	h_inlierIDs = NULL;

	cudaErr = cudaFree(d_curParas);
	d_curParas = NULL;
	cudaErr = cudaFree(d_bestParas);
	d_bestParas = NULL;
	delete[] h_curParas;
	h_curParas = NULL;
	delete[] h_bestParas;
	h_bestParas = NULL;

	cudaErr = cudaFree(d_Hmat);
	d_Hmat = NULL;
	delete[] h_Hmat;
	h_Hmat = NULL;

	cudaErr = cudaFree(d_HypoXs);
	d_HypoXs = NULL;
	cudaErr = cudaFree(d_HypoYs);
	d_HypoYs = NULL;
	cudaErr = cudaFree(d_curHypoys);
	d_curHypoys = NULL;

	delete[] h_HypoXs;
	h_HypoXs = NULL;
	delete[] h_HypoYs;
	h_HypoYs = NULL;
	delete[] h_curHypoys;
	h_curHypoys = NULL;

	cudaErr = cudaFree(d_Amat);
	d_Amat = NULL;
	delete[] h_Amat;
	h_Amat = NULL;

	cudaErr = cudaFree(d_HypoIDs);
	d_HypoIDs = NULL;
	delete[] h_HypoIDs;
	h_HypoIDs = NULL;

	curandDestroyGenerator(gen);
	cudaErr = cudaFree(d_Randf);
	d_Randf = NULL;
	delete[] h_Randf;
	h_Randf = NULL;
	return cudaErr;
}

__global__ void DataDistByGivenModel(double* xvals, double* yvals, size_t pcSize,int paraSize, double* modelPara, double* dists)
{
	int bID = blockIdx.x + blockIdx.y*gridDim.x;
	int tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tID < pcSize)
	{
		double curX = *(xvals + tID), curY = *(yvals + tID);
		double xPows = 1.0, yHat = 0.0;
		double* curDist = dists + tID;
		for (int ii = 0; ii < paraSize; ii++)
		{
			yHat += xPows*modelPara[ii];
			xPows *= curX;
		}
		*curDist = curY - yHat;
	}
}

extern "C" cudaError_t DataFitToGivenModel(__in double* xvals, __in double* yvals, __in size_t pcSize, __in int paraSize, __in double* modelPara, __in double uTh, __in double lTh,
	__out int &resInliers, __out double &modelErr, __out double* &dists)
{
	//Perpare variables
	double* d_xs, *d_ys, *d_Paras, *d_Dists;
	cudaError_t cudaErr;
	unsigned int gridX = (unsigned int)sqrtl((pcSize + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE);
	unsigned int gridY = (unsigned int)((pcSize + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE + gridX - 1) / gridX;
	dim3 blocks(THREAD_SIZE, THREAD_SIZE), grids(gridX, gridY);

	cudaErr = cudaMalloc((void**)&d_xs, sizeof(double)*pcSize);
	cudaErr = cudaMalloc((void**)&d_ys, sizeof(double)*pcSize);
	cudaErr = cudaMalloc((void**)&d_Paras, sizeof(double)*paraSize);
	cudaErr = cudaMalloc((void**)&d_Dists, sizeof(double)*pcSize);
	cudaErr = cudaMemcpy(d_xs, xvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_ys, yvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Paras, modelPara, sizeof(double)*paraSize, cudaMemcpyHostToDevice);
	DataDistByGivenModel<<<grids,blocks>>>(d_xs, d_ys, pcSize, paraSize, d_Paras, d_Dists);
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();
	cudaErr = cudaMemcpy(dists, d_Dists, sizeof(double)*pcSize, cudaMemcpyDeviceToHost);

	int inliers = 0;
	double totalErr = 0.0, tmpE = 0.0, tmpDist = 0.0;
	for (int ii = 0; ii < pcSize; ii++)
	{
		tmpDist = dists[ii];
		tmpE = tmpDist*tmpDist;
		totalErr += tmpE;
		if ((tmpDist > 0 && tmpDist < uTh) || (tmpDist<0 && tmpDist>lTh))
		{
			inliers++;
		}
	}
	resInliers = inliers;
	modelErr = totalErr;

	//Free spaces:
	cudaErr = cudaFree(d_xs);
	cudaErr = cudaFree(d_ys);
	cudaErr = cudaFree(d_Paras);
	cudaErr = cudaFree(d_Dists);
	d_xs = NULL;
	d_ys = NULL;
	d_Paras = NULL;
	d_Dists = NULL;
	return cudaErr;
}

__device__ void FindSpan(__in int n, __in int p,__in double u,__in double* U,__out int* span)
{
	if ((u - U[n + 1] > 0 && u - U[n + 1] < 1e-6) || (u - U[n + 1]<0 && u - U[n + 1]>-1e-6))
	{
		*span = n;
	}
	else
	{
		int low = p, high = n + 1, mid = (low + high) / 2;
		while (u < U[mid] || u >= U[mid + 1])
		{
			if (u < U[mid])
			{
				high = mid;
			}
			else
			{
				low = mid;
			}
			mid = (low + high) / 2;
		}
		*span = mid;
	}
}

__device__ void BasisFuns(__in int i,__in double u,__in int p,__in double* U,__in double* left, __in double* right, __out double* N)
{
	N[0] = 1.0;
	double /*left = new double[p + 1], *right = new double[p + 1],*/ saved, temp;
	for (int j = 1; j <= p; j++)
	{
		left[j] = u - U[i + 1 - j];
		right[j] = U[i + j] - u;
		saved = 0.0;
		for (int r = 0; r < j; r++)
		{
			temp = N[r] / (right[r + 1] + left[j - r]);
			N[r] = saved + right[r + 1] * temp;
			saved = left[j - r] * temp;
		}
		N[j] = saved;
	}
	/*delete[] left;
	left = NULL;
	delete[] right;
	right = NULL;*/
}

__device__ void OneBasisFun(__in int p, __in int m, __in double* U, __in int i, __in double u, __out double* Nip)
{
	if ((i == 0 && u == U[0]) || (i == m - p - 1 && u == U[m]))
	{
		*Nip = 1.0;
	}

	if (u < U[i] || u >= U[i + p + 1])
	{
		*Nip = 0.0;
	}

	double* N = new double[p + 1];
	for (int j = 0; j <= p; j++)
	{
		if (u >= U[i + j] && u < U[i + j + 1]) N[j] = 1.0;
		else N[j] = 0.0;
	}

	double saved = 0.0, Uleft, Uright, temp;
	for (int k = 1; k <= p; k++)
	{
		if (N[0]<1e-6 && N[0]>-1e-6) saved = 0.0;
		else saved = ((u - U[i])*N[0]) / (U[i + k] - U[i]);
		for (int j = 0; j < p - k + 1; j++)
		{
			Uleft = U[i + j + 1];
			Uright = U[i + j + k + 1];
			if (N[j + 1]<1e-6 && N[j + 1]>-1e-6)
			{
				N[j] = saved;
				saved = 0.0;
			}
			else
			{
				temp = N[j + 1] / (Uright - Uleft);
				N[j] = saved + (Uright - u)*temp;
				saved = (u - Uleft)*temp;
			}
		}
	}
	*Nip = N[0];
	delete[] N;
	N = NULL;
}

__device__ void CurvePoint(__in int span,__in int p,__in int n, __in double* N, __in double* Px,__in double* Py,__in double* Wt, __out double *Cx,__out double *Cy)
{
	*Cx = 0.0, *Cy = 0.0;
	//double denom = 0.0;
	for (int ii = 0; ii <= p; ii++)
	{
		if (span - p + ii >= 0 && span - p + ii <= n)
		{
			*Cx += N[ii] * Px[span - p + ii];
			*Cy += N[ii] * Py[span - p + ii];
			/**Cx += N[ii] * Px[span - p + ii] * Wt[span - p + ii];
			*Cy += N[ii] * Py[span - p + ii] * Wt[span - p + ii];
			denom += N[ii] * Wt[span - p + ii];*/
		}		
	}
	/*if (denom<1e-6 && denom>-1e-6)
	{
		denom = 1.0;
	}
	*Cx /= denom;
	*Cy /= denom;*/
}

__global__ void CheckPointOnNURBSCurve(__in double* hypox,__in double* hypoy, __in int hyposize,
	__in double* xvals,__in double* yvals,__in size_t pcSize,__in double Threshold, __out bool* bInOut)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y,
		tIDInB = threadIdx.x + threadIdx.y*blockDim.x;
	if (tID < pcSize)
	{
		int p = 3,             //degree of basis function
			n = hyposize - 1,  //(n+1) control points,id=0,1,...,n.
			m = n + p + 1;     //Size of knot vector(m+1),id=0,1,2,...,m.
		extern __shared__ double U[];//Knot vector
		double* Wt = &U[m+1],
			*CtrPtx = &Wt[n + 1],
			*CtrPty = &CtrPtx[n + 1],
			/**N = &CtrPty[n + 1],
			*left = &N[p + 1],
			*right = &left[p + 1],*/
			uStep = 1.0/512.0,//Searching step along the u parameter in [0,1].
			uVal = 0.0,
			preuVal = 0.0;
		double PCx = xvals[tID], PCy = yvals[tID], curveX, curveY, minDist = -1.0, curDist = -1.0;
		if (tIDInB <= m)
		{
			//Save knot vector(U) with size of m+1 into shared momery.
			if (tIDInB <= p)
			{
				U[tIDInB] = 0.0;
			}
			else if (tIDInB >= m - p)
			{
				U[tIDInB] = 1.0;
			}
			else
			{
				U[tIDInB] = (tIDInB - p)*1.0 / (m - 2 * p)*1.0;
			}
		}

		//Save weights for each control points
		if (tIDInB <= n)
		{
			CtrPtx[tIDInB] = hypox[tIDInB];
			CtrPty[tIDInB] = hypoy[tIDInB];
			Wt[tIDInB] = 1.0;
		}
		__syncthreads();

		//Check inliers for each point
		int span = -1, prespan = -1;
		double* N = (double*)malloc((p + 1) * sizeof(double)),
			*left = (double*)malloc((p + 1) * sizeof(double)),
			*right = (double*)malloc((p + 1) * sizeof(double));
		while (uVal < 1.0)
		{
			FindSpan(n, p, uVal, U, &span);
			if ((preuVal - uVal >= 1e-6 || preuVal - uVal <= -1e-6) || (prespan != span))
			{
				//if (tIDInB == 0)
				//{
					BasisFuns(span, uVal, p, U, left, right, N);
					preuVal = uVal;
					prespan = span;
				//}
				//__syncthreads();
			}
			CurvePoint(span, p, n, N, CtrPtx, CtrPty, Wt, &curveX, &curveY);
			curDist = sqrt((PCx - curveX)*(PCx - curveX) + (PCy - curveY)*(PCy - curveY));
			if (minDist < 0)
			{
				minDist = curDist;
			}
			else if (curDist < minDist)
			{
				minDist = curDist;
			}
			uVal += uStep;
		}

		if (minDist < Threshold)
		{
			bInOut[tID] = true;
		}
		else
		{
			bInOut[tID] = false;
		}

		free(N);
		N = NULL;
		free(left);
		left = NULL;
		free(right);
		right = NULL;
	}
}

__global__ void GetBestNURBSModel(__in bool* bInOut, __in double* xvals, __in double* yvals,__in int ctrptSize, __in unsigned int pcSize,__in int* inliers,
	__inout double* bestCtrX, __inout double* bestCtrY,__inout double* bestRate,__inout int* inlierIDs)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;
	if (tID < 32)
	{
		inliers[tID] = 0;
		for (int ii = 0; ii < pcSize / 32; ii++)
		{
			if (bInOut[tID + ii])
			{
				inliers[tID]++;
			}
		}
		if (tID < pcSize % 32)
		{
			if (bInOut[tID + pcSize / 32 * 32])
			{
				inliers[tID]++;
			}
		}
	}
	__syncthreads();

	if (tID == 0)
	{
		for (int ii = 1; ii < 32; ii++)
		{
			inliers[0] += inliers[ii];
		}
	}
	__syncthreads();

	if (inliers[0] * 1.0 / (pcSize*1.0) > *bestRate)
	{
		if (tID < pcSize)
		{
			if (bInOut[tID])
			{
				inlierIDs[tID] = tID;
			}
			else
			{
				inlierIDs[tID] = -1;
			}
		}

		if (tID < ctrptSize)
		{
			bestCtrX[tID] = xvals[tID];
			bestCtrY[tID] = yvals[tID];
		}

		if (tID == 0)
		{
			*bestRate = inliers[0] * 1.0 / (pcSize*1.0);
		}
	}
}

extern "C" cudaError_t NURBSRANSACOnGPU(__in CtrPtBound inBound, __in double* xvals, __in double* yvals, __in size_t pcSize, __in int maxIters, __in int minInliers, __in double Threshold,
	__out int*& resInliers, __out double &modelErr, __out double* &dists)
{
	cudaError_t cudaErr;
	unsigned int gridX = 1, gridY = 1;
	GetGridFactor((unsigned int)pcSize, (unsigned int)THREAD_SIZE, gridX, gridY);
	dim3 blocks(THREAD_SIZE, THREAD_SIZE), grids(gridX, gridY);
	//Copy total data from host into device:
	double* d_xs, *d_ys;
	cudaErr = cudaMalloc((void**)&d_xs, sizeof(double)*pcSize);
	cudaErr = cudaMalloc((void**)&d_ys, sizeof(double)*pcSize);
	cudaErr = cudaMemcpy(d_xs, xvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_ys, yvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);

	int curIt = 0, bestLast = 0;
	// int bestExit = maxIters*0.1;

	//double bestRate = 0.0;
	float *d_Randf, *h_Randf;
	cudaErr = cudaMalloc((void**)&d_Randf, minInliers * maxIters * sizeof(float));
	h_Randf = new float[minInliers*maxIters];
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, unsigned long long(time(0)));
	curandGenerateUniform(gen, d_Randf, minInliers * maxIters);
	cudaErr = UserDebugCopy(h_Randf, d_Randf, minInliers * maxIters * sizeof(float), cudaMemcpyDeviceToHost);

	size_t *d_HypoIDs, *h_HypoIDs;
	size_t hypoPitch;
	cudaErr = cudaMallocPitch((void**)&d_HypoIDs, &hypoPitch, minInliers * sizeof(size_t), maxIters);
	h_HypoIDs = new size_t[minInliers*maxIters];
	blocks = dim3(32, 32);
	GetGridFactor((unsigned int)(maxIters*minInliers), (unsigned int)32 * 32, grids.x, grids.y);
	ConvertF2IMat << <grids, blocks >> >(d_Randf, minInliers*maxIters, hypoPitch, minInliers, pcSize, d_HypoIDs);
	cudaErr = UserDebugCopy2D(h_HypoIDs, minInliers * sizeof(size_t), d_HypoIDs, hypoPitch, minInliers * sizeof(size_t), maxIters, cudaMemcpyDeviceToHost);

	double *d_HypoXs, *d_HypoYs, *h_HypoXs, *h_HypoYs, *d_curHypoxs, *h_curHypoxs, *d_curHypoys, *h_curHypoys;
	size_t hypoXPitch, hypoYPitch;
	cudaErr = cudaMallocPitch((void**)&d_HypoXs, &hypoXPitch, minInliers * sizeof(double), maxIters);
	cudaErr = cudaMallocPitch((void**)&d_HypoYs, &hypoYPitch, minInliers * sizeof(double), maxIters);
	cudaErr = cudaMalloc((void**)&d_curHypoxs, minInliers * sizeof(double));
	cudaErr = cudaMalloc((void**)&d_curHypoys, minInliers * sizeof(double));
	h_HypoXs = new double[minInliers*maxIters];
	h_HypoYs = new double[minInliers*maxIters];
	h_curHypoxs = new double[minInliers];
	h_curHypoys = new double[minInliers];
	//grids.x = 10, grids.y = 10;
	//GetHypoXYs<<<grids,dim3(28,32)>>>(d_xs, d_ys, d_HypoIDs, hypoPitch, minInliers, maxIters, d_HypoXs, hypoXPitch, d_HypoYs, hypoYPitch);
	//cudaErr = UserDebugCopy2D(h_HypoXs, minInliers * sizeof(double), d_HypoXs, hypoXPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);
	//cudaErr = UserDebugCopy2D(h_HypoYs, minInliers * sizeof(double), d_HypoYs, hypoYPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);
	blocks = dim3(32, 32);
	GetGridFactor((unsigned int)minInliers*maxIters, (unsigned int)32 * 32, grids.x, grids.y);
	GetHypoXYAtOnce << <grids, blocks >> > (d_xs, d_ys, d_HypoIDs, hypoPitch, minInliers, maxIters, d_HypoXs, hypoXPitch, d_HypoYs, hypoYPitch);
	cudaErr = UserDebugCopy2D(h_HypoXs, minInliers * sizeof(double), d_HypoXs, hypoXPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);
	cudaErr = UserDebugCopy2D(h_HypoYs, minInliers * sizeof(double), d_HypoYs, hypoYPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);

	GetGridFactor((unsigned int)maxIters, (unsigned int)32 * 32, grids.x, grids.y);
	SortHypoXY<<<grids,blocks>>>(inBound, minInliers, maxIters, d_HypoXs, hypoXPitch, d_HypoYs, hypoYPitch);
	cudaErr = UserDebugCopy2D(h_HypoXs, minInliers * sizeof(double), d_HypoXs, hypoXPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);
	cudaErr = UserDebugCopy2D(h_HypoYs, minInliers * sizeof(double), d_HypoYs, hypoYPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);

	bool* d_bInOut, *h_bInOut;
	cudaErr = cudaMalloc((void**)&d_bInOut, pcSize * sizeof(bool));
	h_bInOut = new bool[pcSize];

	int* d_inliers, *d_bestIDs, *h_inliers, *h_bestIDs;
	cudaErr = cudaMalloc((void**)&d_inliers, 32 * sizeof(int));
	cudaErr = cudaMalloc((void**)&d_bestIDs, pcSize * sizeof(int));
	h_inliers = new int[32];
	h_bestIDs = new int[pcSize];
	double* d_bestCtrx, *d_bestCtry, *d_bestRate, *h_bestCtrx, *h_bestCtry, *h_bestRate;
	cudaErr = cudaMalloc((void**)&d_bestCtrx, minInliers * sizeof(double));
	cudaErr = cudaMalloc((void**)&d_bestCtry, minInliers * sizeof(double));
	cudaErr = cudaMalloc((void**)&d_bestRate, sizeof(double));
	h_bestCtrx = new double[minInliers];
	h_bestCtry = new double[minInliers];
	h_bestRate = new double(0.0);
	//double preBestRate = -1.0;


	blocks = dim3(16, 16);
	GetGridFactor((unsigned int)pcSize, (unsigned int)blocks.x*blocks.y, grids.x, grids.y);
	while (curIt < maxIters )//&& bestLast < bestExit)
	{
		cudaErr = cudaMemcpy(d_curHypoxs, (double*)((char*)d_HypoXs + curIt*hypoXPitch), minInliers * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaErr = UserDebugCopy(h_curHypoxs, d_curHypoxs, minInliers * sizeof(double), cudaMemcpyDeviceToHost);
		cudaErr = cudaMemcpy(d_curHypoys, (double*)((char*)d_HypoYs + curIt*hypoYPitch), minInliers * sizeof(double), cudaMemcpyDeviceToDevice);
		cudaErr = UserDebugCopy(h_curHypoys, d_curHypoys, minInliers * sizeof(double), cudaMemcpyDeviceToHost);
		
		CheckPointOnNURBSCurve <<<grids, blocks, (4 * minInliers + 3 + 1 )* sizeof(double) >>> (d_curHypoxs, d_curHypoys, minInliers, d_xs, d_ys, pcSize, Threshold, d_bInOut);
		cudaErr = UserDebugCopy(h_bInOut, d_bInOut, pcSize * sizeof(bool), cudaMemcpyDeviceToHost);
		
		GetBestNURBSModel <<<grids, blocks >>> (d_bInOut, d_curHypoxs, d_curHypoys, minInliers, pcSize, d_inliers, d_bestCtrx, d_bestCtry, d_bestRate, d_bestIDs);
		cudaErr = UserDebugCopy(h_inliers, d_inliers, 32 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaErr = UserDebugCopy(h_bestCtrx, d_bestCtrx, minInliers * sizeof(double), cudaMemcpyDeviceToHost);
		cudaErr = UserDebugCopy(h_bestCtry, d_bestCtry, minInliers * sizeof(double), cudaMemcpyDeviceToHost);
		cudaErr = UserDebugCopy(h_bestIDs, d_bestIDs, pcSize * sizeof(int), cudaMemcpyDeviceToHost);
		cudaErr = UserDebugCopy(h_bestRate, d_bestRate, sizeof(double), cudaMemcpyDeviceToHost);
		curIt++;
		/*if (preBestRate < 0.0)
		{
			preBestRate = *h_bestRate;
		}
		else
		{
			if (*h_bestRate > preBestRate)
			{
				preBestRate = *h_bestRate;
				bestLast = 0;
			}
			else
			{
				bestLast++;
			}
		}*/
	}
	
	cudaErr = cudaMemcpy(resInliers, d_bestIDs, pcSize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(&modelErr, d_bestRate, sizeof(double), cudaMemcpyDeviceToHost);

	cudaErr = cudaFree(d_xs);
	d_xs = NULL;
	cudaErr = cudaFree(d_ys);
	d_ys = NULL;

	cudaErr = cudaFree(d_Randf);
	d_Randf = NULL;
	delete[] h_Randf;
	h_Randf = NULL;

	cudaErr = cudaFree(d_HypoIDs);
	d_HypoIDs = NULL;
	delete[] h_HypoIDs;
	h_HypoIDs = NULL;
	
	cudaErr = cudaFree(d_HypoXs);
	d_HypoXs = NULL;
	cudaErr = cudaFree(d_HypoYs);
	d_HypoYs = NULL;
	delete[] h_HypoXs;
	h_HypoXs = NULL;
	delete[] h_HypoYs;
	h_HypoYs = NULL;
	cudaErr = cudaFree(d_curHypoxs);
	d_curHypoxs = NULL;
	cudaErr = cudaFree(d_curHypoys);
	d_curHypoys = NULL;
	delete[] h_curHypoxs;
	h_curHypoxs = NULL;
	delete[] h_curHypoys;
	h_curHypoys = NULL;

	cudaErr = cudaFree(d_bInOut);
	d_bInOut = NULL;
	delete[] h_bInOut;
	h_bInOut = NULL;

	cudaErr = cudaFree(d_inliers);
	d_inliers = NULL;
	cudaErr = cudaFree(d_bestIDs);
	d_bestIDs = NULL;
	cudaErr = cudaFree(d_bestCtrx);
	d_bestCtrx = NULL;
	cudaErr = cudaFree(d_bestCtry);
	d_bestCtry = NULL;
	cudaErr = cudaFree(d_bestRate);
	d_bestRate = NULL;

	delete[] h_inliers;
	h_inliers = NULL;
	delete[] h_bestIDs;
	h_bestIDs = NULL;
	delete[] h_bestCtrx;
	h_bestCtrx = NULL;
	delete[] h_bestCtry;
	h_bestCtry = NULL;
	delete h_bestRate;
	h_bestRate = NULL;

	return cudaErr;
}