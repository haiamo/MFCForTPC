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

#define CHECK_D_ZERO(x) (abs(x))<1e-8
#define THREAD_SIZE 16

#define GETGRIDX(dtLen)  sqrtl(((dtLen) + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE)
#define GETGRIDY(dtLen, x)  (((dtLen) + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE + (x) - 1) / (x)

__global__ void SetupMatrices(double* xvals, int maxIt, cudaExtent Aextent, cudaPitchedPtr Amat)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		double* curA = (double*)((char*)Amat.ptr + tid*Amat.pitch*Aextent.height);
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
					curA[curID] = curA[preLineID] * xvals[tid*Arows + rowi];
				}
			}
		}
	}
}

__global__ void GetHypoModelPara(double* yvals,int maxIt, size_t pcSize, cudaPitchedPtr Amat, cudaExtent Aextent,
	size_t colvPitch, double* colVecs, size_t paraPitch, double* paras)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		double* curA = (double*)((char*)Amat.ptr + tid*Amat.pitch*Aextent.height);//Column major storage
		double* curYs = yvals + tid*pcSize;
		double *outPara = (double*)((char*)paras + tid*paraPitch), tmpVal = 0.0;

		int curID = 0;
		double vnorm = 0.0, sigma = 0.0, beta = 0.0, ytv = 0.0;
		int Arows = Aextent.width / sizeof(double), Acols = Aextent.height, ApitchLen = Amat.pitch / sizeof(double);
		double* col_vec = (double*)((char*)colVecs + tid*colvPitch), *Atv_vec = (double*)malloc(sizeof(double)*Acols);
		
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
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		double* curPara = (double*)((char*)paras+tid*paraPitch), *outErr = modelErr + tid, *outDist = (double*)((char*)dists + tid*distPitch);
		*outErr = 0.0;
		bool *curbInOut = (bool*)((char*)bInOut + tid*bInOutPitch);
		int *curInlNum = inlierNum + tid;
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
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		int curInliers = *(inlierNum + tid);
		if (curInliers > paraSize)
		{
			bool* curInOrOut = (bool*)((char*)bInOrOut + tid*bInOutPitch);
			double* curA = (double*)((char*)Amat.ptr + tid*Amat.pitch*Aextent.height);
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
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		int Arows = *(inlierNum + tid), Acols = paraSize;
		if (Arows > Acols)
		{
			bool* curbInlier = (bool*)((char*)bInOut + tid*bInOutPitch);
			double* curA = (double*)((char*)Amat.ptr + tid*Amat.pitch*Aextent.height);
			double *outPara = (double*)((char*)paras+tid*paraPitch), tmpVal = 0.0;

			int curID = 0;
			double vnorm = 0.0, sigma = 0.0, beta = 0.0, ytv = 0.0;
			double* col_vec = (double*)((char*)colVecs + tid*colVPitch), *Atv_vec = (double*)((char*)Atvs+tid*Atv_pitch), *curYs = (double*)((char*)allYs + tid*allYsPitch);
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
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		int Arows = *(inlierNum + tid), Acols = paraSize;
		if (Arows > Acols)
		{
			double* curPara = (double*)((char*)paras + tid*paraPitch), *outErr = modelErr + tid, *outDist = (double*)((char*)dists + tid*distPitch);
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

extern "C" cudaError_t RANSACOnGPU(double* xvals, double* yvals, size_t pcSize, int maxIters, int minInliers, int paraSize, double uTh, double lTh,
	double* &paraList, int* &resInliers, double* &modelErr, double* &dists, int &resIters)
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
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < fLen)
	{
		size_t* pLine = (size_t*)((char*)uiVals + (tid / width)*uiPitch);
		if (fVals[tid] < 1e-6)
		{
			uiVals[tid] = 0;
		}
		else
		{
			pLine[tid%width] = size_t((fVals[tid] - 1e-6) / (1 - 1e-6)*uiMax);
		}
	}
}

__global__ void GetHypoXYs(double* xs, double* ys, size_t* hypoID2D, size_t idPitch, size_t width, size_t height, double* hypoxs, size_t xPitch, double* hypoys, size_t yPitch)
{
	unsigned int blockID = blockIdx.y*gridDim.x + blockIdx.x,
		threadID = blockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x,
		threadColID = blockIdx.x*blockDim.x + threadIdx.x,
		threadRowID = blockIdx.y*blockDim.y + threadIdx.y,
		gThWidth = gridDim.x*blockDim.x, gThHeight = gridDim.y*blockDim.y;
	unsigned int totalThreads = gridDim.x*gridDim.y*blockDim.x*blockDim.y,
		loopTimes = (width*height + totalThreads - 1) / totalThreads,
		threadLimit = totalThreads;
	unsigned int blockValColLimit = ((width + blockDim.x - 1) / blockDim.x < gridDim.x) ? (width + blockDim.x - 1) / blockDim.x : gridDim.x,
		blockValRowLimit = ((height + blockDim.y - 1) / blockDim.y < gridDim.y) ? (height + blockDim.y - 1) / blockDim.y : gridDim.y;

	if (blockIdx.x < blockValColLimit && blockIdx.y < blockValRowLimit)
	{
		unsigned int tColMove = (width + gThWidth - 1) / gThWidth, tRowMove = (height + gThHeight - 1) / gThHeight,
			curCol = 0, curRow = 0;
		for (unsigned int colMvI = 0; colMvI < tColMove; colMvI++)
		{
			for (unsigned int rowMvI = 0; rowMvI < tRowMove; rowMvI++)
			{
				curCol = colMvI*gThWidth + threadColID;
				curRow = rowMvI*gThHeight + threadRowID;
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
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	size_t* curHypoLine = NULL;
	if (lineID < height)
	{
		curHypoLine = (size_t*)((char*)hypoID2D + lineID*linePitch);
	}
	if (NULL != curHypoLine)
	{
		extern __shared__ double curX[];
		int tidInBlock = threadIdx.y*blockDim.x + threadIdx.x;
		unsigned int blockLen = blockDim.x*blockDim.y;
		if (blockLen >= width)
		{
			if (tidInBlock < width)
			{
				curX[tidInBlock] = xs[curHypoLine[tidInBlock]];
			}
		}
		else
		{
			for (unsigned int ii = 0; ii < blockLen / width; ii++)
			{
				curX[ii*blockLen + tidInBlock] = xs[curHypoLine[ii*blockLen + tidInBlock]];
			}
			if (tidInBlock < blockLen%width)
			{
				curX[blockLen / width*blockLen + tidInBlock] = xs[curHypoLine[blockLen / width*blockLen + tidInBlock]];
			}
		}

		if (tid < width*paraSize)
		{
			int row = 0, col = 0;
			row = int(tid / paraSize);
			col = tid%paraSize;
			double tmpVal = 1.0;
			for (int ii = 1; ii <= col; ii++)
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
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < n)
	{
		double* ioMatRow = (double*)((char*)inoutMat + tid*matPitch);
		ioMatRow[tid] = 1.0;
	}
}

__global__ void GetHk(size_t colID, double* Amat, size_t APitch, size_t widthA, size_t heightA, double* Hmat, size_t HPitch)
{
	unsigned int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	unsigned int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	unsigned int needBlocks = 0;
	unsigned int blockLen = blockDim.x*blockDim.y;

	//needBlocks = (heightA*heightA + blockLen - 1) / blockLen;
	needBlocks = ((heightA + blockDim.x - 1) / blockDim.x)*((heightA + blockDim.y - 1) / blockDim.y);

	if (blockIdx.x < (heightA + blockDim.x - 1) / blockDim.x && blockIdx.y < (heightA + blockDim.y - 1) / blockDim.y)
	{
		extern __shared__ double inShared[];
		double* colVec = (double*)inShared, *pBeta = (double*)&colVec[heightA];
		int tidInBlock = threadIdx.y*blockDim.x + threadIdx.x;
		unsigned int blockLen = blockDim.x*blockDim.y;

		double* curALine;
		if (blockLen >= heightA)
		{
			if (tidInBlock < heightA)
			{
				curALine = (double*)((char*)Amat + tidInBlock*APitch);
				colVec[tidInBlock] = curALine[colID];
			}
		}
		else
		{
			for (unsigned int ii = 0; ii < heightA / blockLen; ii++)
			{
				curALine = (double*)((char*)Amat + (ii*blockLen + tidInBlock)*APitch);
				colVec[ii*blockLen + tidInBlock] = curALine[colID];
			}
			if (tidInBlock < heightA%blockLen)
			{
				curALine = (double*)((char*)Amat + (blockLen* (heightA / blockLen) + tidInBlock)*APitch);
				colVec[blockLen* (heightA / blockLen) + tidInBlock] = curALine[colID];
			}
		}

		__syncthreads();

		double sigma = 0.0, beta = 0.0, norm = 0.0;
		if (tidInBlock == 0)
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
	unsigned int blockID = blockIdx.x + gridDim.x*blockIdx.y,
		threadRowID = blockIdx.y*blockDim.y + threadIdx.y, threadColID = blockIdx.x*blockDim.x + threadIdx.x,
		threadInBlockID = threadIdx.y*blockDim.x + threadIdx.x;
	unsigned int  LTileRow = 0, LTileCol = 0, RTileRow = 0, RTileCol = 0;
	unsigned int sharedTileID = 0, globalTileID = 0;

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
				sharedTileID = threadIdx.y*LTileCol + ii*blockDim.x + threadIdx.x;
				globalTileID = threadIdx.y*LPitch / sizeof(double) + ii*blockDim.x + threadIdx.x;
				LTile[sharedTileID] = inTile[globalTileID];
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
				sharedTileID = ii*RTileCol*blockDim.y + threadIdx.y*RTileCol + threadIdx.x;
				globalTileID = (ii*blockDim.y + threadIdx.y)*RPitch / sizeof(double) + blockIdx.x*blockDim.x + threadIdx.x;
				RTile[sharedTileID] = inTile[globalTileID];
			}
		}
		__syncthreads();

		if (threadRowID<LRows && threadColID<RCols)
		{
			double sum = 0.0;

			for (unsigned int ii = 0; ii < LCols; ii++)
			{
				//Modified for Rows and columns.
				sum += LTile[threadIdx.y*LCols + ii] * RTile[ii*RTileCol + threadIdx.x];
			}

			double *resLine = (double*)((char*)ResMat + threadRowID*ResPitch);
			if (CHECK_D_ZERO(sum))
			{
				resLine[threadColID] = 0.0;
			}
			else
			{
				resLine[threadColID] = sum;
			}	
		}
	}
}

__global__ void MatTranspose(double* inMat, size_t inPitch, size_t inRows, size_t inCols, double* outMat, size_t outPitch)
{
	unsigned int blockID = blockIdx.y*gridDim.x + blockIdx.x,
		threadID = blockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x,
		threadColID = blockIdx.x*blockDim.x + threadIdx.x,
		threadRowID = blockIdx.y*blockDim.y + threadIdx.y,
		gThWidth = gridDim.x*blockDim.x, gThHeight = gridDim.y*blockDim.y;
	unsigned int totalThreads = gridDim.x*gridDim.y*blockDim.x*blockDim.y,
		loopTimes = (inCols*inRows + totalThreads - 1) / totalThreads,
		threadLimit = totalThreads;
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
				curCol = colMvI*gThWidth + threadColID;
				curRow = rowMvI*gThHeight + threadRowID;
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
		blockID = blockIdx.x + blockIdx.y*gridDim.x;

	if (blockID < needBlocks)
	{
		unsigned int threadIDInBlock = threadIdx.x + threadIdx.y*blockDim.x, tidLimit = blockDim.x*blockDim.y;
		if ((blockID + 1)*blockDim.x*blockDim.y > matRows)
		{
			tidLimit = matRows % (blockDim.x*blockDim.y);
		}

		if (threadIDInBlock < tidLimit)
		{
			double* matTileLine = (double*)((char*)inMat + threadIDInBlock*matPitch);
			for (unsigned int ii = 0; ii < matCols; ii++)
			{
				matTile[threadIDInBlock*matCols + ii] = matTileLine[ii];
			}
		}

		if (threadIDInBlock < matCols)
		{
			vecTile[threadIDInBlock] = inVec[threadIDInBlock];
		}

		__syncthreads();

		if (threadIDInBlock < tidLimit)
		{
			double tmpVal = 0.0;
			for (unsigned int ii = 0; ii < matCols; ii++)
			{
				tmpVal += matTile[threadIDInBlock*matCols + ii] * vecTile[ii];
			}
			outVec[blockID*blockDim.x*blockDim.y + threadIDInBlock] = tmpVal;
		}
	}
}


__global__ void SolveRMatSystem(double* Rmat, size_t RPitch, size_t valLen, double* inYs, double* outRes)
{
	unsigned int blockID = blockIdx.y*gridDim.x + blockIdx.x,
		threadID = blockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;

	if (threadID == 0)
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

extern "C" cudaError_t RANSACOnGPU1(double* xvals, double* yvals, size_t pcSize, int maxIters, int minInliers, int paraSize, double uTh, double lTh,
	double* &bestParas, int* &resInliers, double &modelErr, double* &dists)
{
	cudaError_t cudaErr;
	unsigned int gridX = (unsigned int)GETGRIDX(maxIters);
	unsigned int gridY = (unsigned int)GETGRIDY(maxIters, gridX);
	dim3 blocks(THREAD_SIZE, THREAD_SIZE), grids(gridX, gridY);
	//Copy total data from host into device:
	double* d_xs, *d_ys;
	cudaErr = cudaMalloc((void**)&d_xs, sizeof(double)*pcSize);
	cudaErr = cudaMalloc((void**)&d_ys, sizeof(double)*pcSize);
	cudaErr = cudaMemcpy(d_xs, xvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_ys, yvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);

	int curIt = 0, bestLast = 0;
	const int bestExit = maxIters*0.1;

	double curRate = 0.0, bestRate = 0.0;
	float *d_Randf, *h_Randf;
	cudaErr = cudaMalloc((void**)&d_Randf, minInliers * maxIters * sizeof(float));
	h_Randf = new float[minInliers*maxIters];
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, unsigned long long(time(0)));
	curandGenerateUniform(gen, d_Randf, minInliers * maxIters);
	cudaErr = cudaMemcpy(h_Randf, d_Randf, minInliers * maxIters * sizeof(float), cudaMemcpyDeviceToHost);

	size_t *d_HypoIDs, *h_HypoIDs;
	size_t hypoPitch;
	cudaErr = cudaMallocPitch((void**)&d_HypoIDs, &hypoPitch, minInliers * sizeof(size_t), maxIters);
	h_HypoIDs = new size_t[minInliers*maxIters];
	grids.x = (unsigned int)GETGRIDX(maxIters*minInliers);
	grids.y = (unsigned int)GETGRIDY(maxIters*minInliers, grids.x);
	ConvertF2IMat<<<grids,blocks>>>(d_Randf, minInliers*maxIters, hypoPitch, minInliers, pcSize, d_HypoIDs);
	cudaErr = cudaMemcpy2D(h_HypoIDs, minInliers * sizeof(size_t), d_HypoIDs, hypoPitch, minInliers * sizeof(size_t), maxIters, cudaMemcpyDeviceToHost);
	
	double *d_HypoXs, *d_HypoYs, *h_HypoXs, *h_HypoYs, *d_curHypoys, *h_curHypoys;
	size_t hypoXPitch, hypoYPitch;
	cudaErr = cudaMallocPitch((void**)&d_HypoXs, &hypoXPitch, minInliers * sizeof(double), maxIters);
	cudaErr = cudaMallocPitch((void**)&d_HypoYs, &hypoYPitch, minInliers * sizeof(double), maxIters);
	cudaErr = cudaMalloc((void**)&d_curHypoys, minInliers * sizeof(double));
	h_HypoXs = new double[minInliers*maxIters];
	h_HypoYs = new double[minInliers*maxIters];
	h_curHypoys = new double[minInliers];
	grids.x = 10, grids.y = 10;
	GetHypoXYs<<<grids,blocks>>>(d_xs, d_ys, d_HypoIDs, hypoPitch, minInliers, maxIters, d_HypoXs, hypoXPitch, d_HypoYs, hypoYPitch);
	cudaErr = cudaMemcpy2D(h_HypoXs, minInliers * sizeof(double), d_HypoXs, hypoXPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy2D(h_HypoYs, minInliers * sizeof(double), d_HypoYs, hypoYPitch, minInliers * sizeof(double), maxIters, cudaMemcpyDeviceToHost);
	
	double* d_Amat, *h_Amat;
	size_t APitch;
	cudaErr = cudaMallocPitch((void**)&d_Amat, &APitch, paraSize * sizeof(double), minInliers);
	h_Amat = new double[minInliers*paraSize];

	double* d_Hmat, *d_Qmat, *d_Rmat, *d_Qtmat;
	size_t HPitch, QPitch, RPitch, QtPitch;
	cudaErr = cudaMallocPitch((void**)&d_Hmat, &HPitch, minInliers * sizeof(double), minInliers);
	cudaErr = cudaMallocPitch((void**)&d_Qmat, &QPitch, minInliers * sizeof(double), minInliers);
	cudaErr = cudaMallocPitch((void**)&d_Rmat, &RPitch, paraSize * sizeof(double), minInliers);
	cudaErr = cudaMallocPitch((void**)&d_Qtmat, &QtPitch, minInliers * sizeof(double), minInliers);


	double*h_Hmat, *h_Qmat, *h_Rmat, *h_Qtmat;
	h_Hmat = new double[minInliers*minInliers];
	h_Qmat = new double[minInliers*minInliers];
	h_Rmat = new double[minInliers*paraSize];
	h_Qtmat = new double[minInliers*minInliers];

	double* d_curParas, *h_curParas;
	cudaErr = cudaMalloc((void**)&d_curParas, paraSize * sizeof(double));
	h_curParas = new double[paraSize];
	while (curIt < maxIters && bestLast < bestExit)
	{
		//Generate A and y by hypo-inliers
		cudaErr = cudaMemset2D(d_Amat, APitch, 0.0, paraSize * sizeof(double), minInliers);
		grids.x = (unsigned int)GETGRIDX(maxIters);
		grids.y = (unsigned int)GETGRIDY(maxIters, grids.x);
		blocks = dim3(THREAD_SIZE, THREAD_SIZE, 1);
		SetupSingleMatrix<<<grids, blocks, minInliers*sizeof(double)>>>(d_xs, d_HypoIDs, hypoPitch, minInliers, maxIters, curIt, paraSize, APitch, d_Amat);
		cudaErr = cudaMemcpy2D(h_Amat, paraSize * sizeof(double), d_Amat, APitch, paraSize * sizeof(double), minInliers, cudaMemcpyDeviceToHost);
		cudaErr = cudaMemcpy2D(d_Rmat, RPitch, d_Amat, APitch, paraSize * sizeof(double), minInliers, cudaMemcpyDeviceToDevice);
		cudaErr = cudaMemcpy2D(h_Rmat, paraSize * sizeof(double), d_Rmat, RPitch, paraSize * sizeof(double), minInliers, cudaMemcpyDeviceToHost);

		//Solve A*alpha=y on hypo-inliers 
		grids.x = (unsigned int)GETGRIDX(minInliers);
		grids.y = (unsigned int)GETGRIDY(minInliers, grids.x);
		cudaErr = cudaMemset2D(d_Hmat, HPitch, 0.0, minInliers * sizeof(double), minInliers);
		SetIdentityMat <<<grids, blocks >>> (d_Hmat, HPitch, minInliers);
		cudaErr = cudaMemcpy2D(h_Hmat, minInliers * sizeof(double), d_Hmat, HPitch, minInliers * sizeof(double), minInliers, cudaMemcpyDeviceToHost);
		
		cudaErr = cudaMemset2D(d_Qmat, QPitch, 0.0, minInliers * sizeof(double), minInliers);
		SetIdentityMat <<<grids, blocks >>> (d_Qmat, QPitch, minInliers);
		cudaErr = cudaMemcpy2D(h_Qmat, minInliers * sizeof(double), d_Qmat, QPitch, minInliers * sizeof(double), minInliers, cudaMemcpyDeviceToHost);

		//A(k+1)=H(k)*A(k), Q(k+1)=H(k+1)*Q(k)
		//Finally,it obtains R=A(n) and Q=Q(n),while the linear system A*alpha=y is converted to R*alpha=Q^t*y.
		blocks = dim3(8, 8, 1);
		for (size_t ii = 0; ii < paraSize; ii++)
		{
			grids.x = (minInliers + blocks.x - 1) / blocks.x;
			grids.y = (minInliers + blocks.y - 1) / blocks.y;
			GetHk<<<grids, blocks, (minInliers + 1) * sizeof(double)>>>(ii, d_Rmat, RPitch, paraSize, minInliers, d_Hmat, HPitch);
			cudaErr = cudaMemcpy2D(h_Hmat, minInliers * sizeof(double), d_Hmat, HPitch, minInliers * sizeof(double), minInliers, cudaMemcpyDeviceToHost);
			grids.x = (minInliers + blocks.x - 1) / blocks.x;
			grids.y = (minInliers + blocks.y - 1) / blocks.y;
			MatMultMat<<<grids, blocks, minInliers*(blocks.x + blocks.y) * sizeof(double)>>> (d_Hmat, HPitch, minInliers, minInliers, d_Qmat, QPitch, minInliers, minInliers, d_Qmat, QPitch);
			cudaErr = cudaMemcpy2D(h_Qmat, minInliers * sizeof(double), d_Qmat, QPitch, minInliers * sizeof(double), minInliers, cudaMemcpyDeviceToHost);
			grids.x = (paraSize + blocks.x - 1) / blocks.x;
			grids.y = (minInliers + blocks.y - 1) / blocks.y;
			MatMultMat<<<grids, blocks, minInliers*(blocks.x + blocks.y) * sizeof(double)>>> (d_Hmat, HPitch, minInliers, minInliers, d_Rmat, RPitch, minInliers, paraSize, d_Rmat, RPitch);
			cudaErr = cudaMemcpy2D(h_Rmat, paraSize * sizeof(double), d_Rmat, RPitch, paraSize * sizeof(double), minInliers, cudaMemcpyDeviceToHost);
		}

		//MatTranspose<<<grids,blocks>>>(d_Qmat, QPitch, minInliers, minInliers, d_Qtmat, QtPitch);
		//cudaErr = cudaMemcpy2D(h_Qtmat, minInliers * sizeof(double), d_Qtmat, QtPitch, minInliers * sizeof(double), minInliers, cudaMemcpyDeviceToHost);
		
		blocks.x = 4, blocks.y = 4;
		grids.x = (minInliers + blocks.x*blocks.y - 1) / (blocks.x*blocks.y);
		grids.y = 1;	
		MatMultVec<<<grids,blocks,minInliers*(blocks.x*blocks.y+1)*sizeof(double)>>>(d_Qmat, QPitch, minInliers, minInliers, d_HypoYs+curIt*hypoYPitch/sizeof(double),d_curHypoys);
		cudaErr = cudaMemcpy(h_curHypoys, d_curHypoys, minInliers * sizeof(double), cudaMemcpyDeviceToHost);

		SolveRMatSystem<<<1,1>>>(d_Rmat, RPitch, paraSize, d_curHypoys, d_curParas);
		cudaErr = cudaMemcpy(h_curParas, d_curParas, paraSize * sizeof(double), cudaMemcpyDeviceToHost);

		//Check inliers on total data
		curIt++;
	}

	cudaErr = cudaFree(d_curParas);
	d_curParas = NULL;
	delete[] h_curParas;
	h_curParas = NULL;

	cudaErr = cudaFree(d_Hmat);
	d_Hmat = NULL;
	delete[] h_Hmat;
	h_Hmat = NULL;
	cudaErr = cudaFree(d_Qmat);
	d_Qmat = NULL;
	delete[] h_Qmat;
	h_Qmat = NULL;
	cudaErr = cudaFree(d_Rmat);
	d_Rmat = NULL;
	delete[] h_Rmat;
	h_Rmat = NULL;
	cudaErr = cudaFree(d_Qtmat);
	d_Qtmat = NULL;
	delete[] h_Qtmat;
	h_Qmat = NULL;

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
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < pcSize)
	{
		double curX = *(xvals + tid), curY = *(yvals + tid);
		double xPows = 1.0, yHat = 0.0;
		double* curDist = dists + tid;
		for (int ii = 0; ii < paraSize; ii++)
		{
			yHat += xPows*modelPara[ii];
			xPows *= curX;
		}
		*curDist = curY - yHat;
	}
}

extern "C" cudaError_t DataFitToGivenModel(double* xvals, double* yvals, size_t pcSize,int paraSize, double* modelPara, double uTh, double lTh,
	int &resInliers, double &modelErr, double* &dists)
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