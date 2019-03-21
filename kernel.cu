#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "cuda.h"
#include "device_functions.h"
#include "cublas.h"
#include "cublas_v2.h"
#include "cudaMain.h"
#include "stdio.h"

#define CHECK_D_ZERO(x) (x<1e-8 && x>-1e-8)
#define THREAD_SIZE 16

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
		size_t inlListNum = 0;
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
	cudaErr = cudaDeviceSynchronize();
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
	cudaErr = cudaDeviceSynchronize();
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
	cudaErr = cudaDeviceSynchronize();
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
	int valuableIts = 0, curID = 0;
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
	cudaErr = cudaDeviceSynchronize();
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
	cudaErr = cudaDeviceSynchronize();
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
	cudaErr = cudaDeviceSynchronize();
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

__global__ void DataDistByGivenModel(double* xvals, double* yvals, size_t pcSize,int paraSize, double* modelPara, double* dists)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < pcSize)
	{
		double curX = *(xvals + tid), curY = *(yvals + tid);
		double xPows = 1.0, yHat = 0.0;
		double* curModelPara = modelPara;
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
	int* &resInliers, double* &modelErr)
{
	//Perpare variables
	double* d_xs, *d_ys, *d_Paras, *d_Dists, *h_Dists;
	cudaError_t cudaErr;
	unsigned int gridX = (unsigned int)sqrtl((pcSize + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE);
	unsigned int gridY = (unsigned int)((pcSize + THREAD_SIZE*THREAD_SIZE - 1) / THREAD_SIZE / THREAD_SIZE + gridX - 1) / gridX;
	dim3 blocks(THREAD_SIZE, THREAD_SIZE), grids(gridX, gridY);

	cudaErr = cudaMalloc((void**)&d_xs, sizeof(double)*pcSize);
	cudaErr = cudaMalloc((void**)&d_ys, sizeof(double)*pcSize);
	cudaErr = cudaMalloc((void**)&d_Paras, sizeof(double)*paraSize);
	cudaErr = cudaMalloc((void**)&d_Dists, sizeof(double)*pcSize);
	h_Dists = (double*)malloc(sizeof(double)*pcSize);
	cudaErr = cudaMemcpy(d_xs, xvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_ys, yvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Paras, modelPara, sizeof(double)*paraSize, cudaMemcpyHostToDevice);
	DataDistByGivenModel<<<grids,blocks>>>(d_xs, d_ys, pcSize, paraSize, d_Paras, d_Dists);
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();
	cudaErr = cudaMemcpy(h_Dists, d_Dists, sizeof(double)*pcSize, cudaMemcpyDeviceToHost);

	int inliers = 0;
	double totalErr = 0.0, tmpE = 0.0, tmpDist = 0.0;
	for (int ii = 0; ii < pcSize; ii++)
	{
		tmpDist = h_Dists[ii];
		tmpE = tmpDist*tmpDist;
		totalErr += tmpE;
		if ((tmpDist > 0 && tmpDist < uTh) || (tmpDist<0 && tmpDist>lTh))
		{
			inliers++;
		}
	}
	*resInliers = inliers;
	*modelErr = totalErr;
	return cudaErr;
}