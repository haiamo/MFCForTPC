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

__global__ void SetupMatrices(double* xvals, int maxIt, cudaPitchedPtr Amat, cudaExtent Aextent)
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

__global__ void GetHypoModelPara(double* yvals,int maxIt, size_t pcsize, int parasize, cudaPitchedPtr Amat, cudaExtent Aextent,
	size_t colvPitch, double* colVecs, size_t paraPitch, double* paras)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		double* curA = (double*)((char*)Amat.ptr + tid*Amat.pitch*Aextent.height);//Column major storage
		double* curYs = yvals + tid*pcsize;
		double *outPara = (double*)((char*)paras + tid*paraPitch), tmpVal = 0.0;

		int curID = 0;
		double vnorm = 0.0, sigma = 0.0, beta = 0.0, ytv = 0.0;
		int Arows = Aextent.width / sizeof(double), Acols = Aextent.height, ApitchLen = Amat.pitch / sizeof(double);
		double* col_vec = (double*)((char*)colVecs + tid*colvPitch), *Atv_vec = (double*)malloc(sizeof(double)*Acols);
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

__global__ void CheckPointInOrOut(double* xs, double* ys, size_t pcsize, int maxIt, int parasize,double uTh,double lTh, size_t paraPitch, double* paras,
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
		for (size_t ii = 0; ii < pcsize; ii++)
		{
			xPows = 1.0;
			yHat = 0.0;
			curbInOut[ii] = false;
			for (int parii = 0; parii < parasize; parii++)
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
		tmpErr /= pcsize;
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

__global__ void GetModelParawithVariateAs(double* yvals, size_t pcsize,int maxIt,int paraSize, cudaExtent Aextent, cudaPitchedPtr Amat, int* inlierNum,
	  size_t colVPitch, double* colVecs, size_t allYsPitch, double* allYs, size_t bInOutPitch, bool* bInOut, size_t paraPitch, double* paras)
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
			double* col_vec = (double*)((char*)colVecs + tid*colVPitch), *Atv_vec = (double*)malloc(sizeof(double)*Acols), *curYs = (double*)((char*)allYs + tid*allYsPitch);
			size_t ApitchLen = Amat.pitch / sizeof(double);
			
			//Prepared the y column vector.
			for (size_t rowi = 0; rowi < pcsize; rowi++)
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
			free(Atv_vec);
			Atv_vec = NULL;
		}
	}
}

__global__ void GetModelSqDist(double* xs, double* ys,size_t pcsize, int maxIt, int parasize, size_t paraPitch, double* paras, int* inlierNum,
							 double* modelErr,size_t distPitch, double* dists)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		int Arows = *(inlierNum + tid), Acols = parasize;
		if (Arows > Acols)
		{
			double* curPara = (double*)((char*)paras + tid*paraPitch), *outErr = modelErr + tid, *outDist = (double*)((char*)dists + tid*distPitch);
			double yHat = 0.0;//y^^=paras[k]*x^(k-1)+...paras[0], k=order of fitted polynomial
			double xPows = 0.0, tmpDist = 0.0, tmpErr = 0.0;
			*outErr = 0.0;
			for (size_t ii = 0; ii < pcsize; ii++)
			{
				xPows = 1.0;
				yHat = 0.0;
				for (int parii = 0; parii < parasize; parii++)
				{
					yHat += xPows*curPara[parii];
					xPows *= xs[ii];
				}
				tmpDist = ys[ii] - yHat;
				outDist[ii] = tmpDist;
				tmpErr += tmpDist*tmpDist;
			}
			tmpErr /= pcsize;
			*outErr = tmpErr;
		}
	}
}

extern "C" cudaError_t RANSACOnGPU(double* xvals, double* yvals, size_t pcsize, int maxIters, int minInliers, int parasize, double uTh, double lTh,
	double* &hst_hypox, double* &hst_hypoy, double* &As, double** &Qs, double** &taus, double** &Rs, double**& paras,
	double* &modelErr, double* &dists, int &hypoIters)
{
	//Prepare device space and parameters.
	cudaError_t cudaErr;
	double *dev_hypox = NULL, *dev_hypoy = NULL;
	unsigned int *hst_hypoIDs;
	hst_hypoIDs = (unsigned int *)malloc(sizeof(unsigned int)*maxIters*minInliers);

	//Generate random hypo-inliers IDs.
	srand(unsigned(time(NULL)));
	for (size_t ii = 0; ii < maxIters*minInliers; ii++)
	{
		hst_hypoIDs[ii] = rand() % pcsize;
		hst_hypox[ii] = xvals[hst_hypoIDs[ii]];
		hst_hypoy[ii] = yvals[hst_hypoIDs[ii]];
	}

	cudaErr = cudaMalloc((void**)&dev_hypox, sizeof(double)*maxIters*minInliers);
	cudaErr = cudaMalloc((void**)&dev_hypoy, sizeof(double)*maxIters*minInliers);

	cudaErr = cudaMemcpy(dev_hypox, hst_hypox, sizeof(double)*maxIters*minInliers, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(dev_hypoy, hst_hypoy, sizeof(double)*maxIters*minInliers, cudaMemcpyHostToDevice);

	/*********************
	Get the 2D curve model by Least Square Method with QR Decomposition
	Sove the equation A*a=y, where A has the size n*m, a has m*1 and y has n*1.
	In this problem, n(rows) is the size of hypo-inliers Points, m(cols) equals to the order of fitted ploynomial plus one(m=order+1).
	For parallel computing, there are MaxIteration(mi:num) matrices in total.
	All matrices is column-major stored continuously.
	**********************/

	int rows = minInliers, cols = parasize;
	int num = maxIters, ltau = cols;//Number of matrices, known as the maximum iterations of RANSAC
	double *h_Amat=As;
	cudaPitchedPtr d_Amat;
	cudaExtent Aextent;
	cudaMemcpy3DParms AcpParm = { 0 };
	Aextent = make_cudaExtent(sizeof(double)*rows, cols, num);
	cudaErr = cudaMalloc3D(&d_Amat, Aextent);
	SetupMatrices <<<(maxIters + 255) / 256, 256 >>> (dev_hypox, maxIters, d_Amat, Aextent);
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();
	AcpParm.dstPtr = make_cudaPitchedPtr((void*)h_Amat, sizeof(double)*rows, rows, cols);
	AcpParm.srcPtr = d_Amat;
	AcpParm.extent = Aextent;
	AcpParm.kind = cudaMemcpyDeviceToHost;
	cudaErr = cudaMemcpy3D(&AcpParm);

	//__global__ void GetHypoModelPara(double* yvals, int maxIt, size_t pcsize, int parasize, cudaPitchedPtr Amat, cudaExtent Aextent,
		//size_t colvPitch, double** colVecs, size_t paraPitch, double* paras)
	double* d_ColVec, *d_Paras, *h_Paras;
	size_t colVecPitch, parasPitch;
	h_Paras = (double*)malloc(sizeof(double)*maxIters*parasize);
	cudaErr = cudaMallocPitch((void**)&d_ColVec, &colVecPitch, sizeof(double)*minInliers, maxIters);
	cudaErr = cudaMallocPitch((void**)&d_Paras, &parasPitch, sizeof(double)*parasize, maxIters);
	cudaErr = cudaMemset2D(d_ColVec, colVecPitch, 0, sizeof(double)*minInliers, maxIters);
	cudaErr = cudaMemset2D(d_Paras, parasPitch, 0, sizeof(double)*parasize, maxIters);
	GetHypoModelPara <<<(maxIters + 255) / 256, 256 >>> (dev_hypoy, maxIters, minInliers, parasize, d_Amat, Aextent,
		colVecPitch, d_ColVec, parasPitch, d_Paras);
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();
	cudaErr = cudaMemcpy2D(h_Paras, sizeof(double)*parasize, d_Paras, parasPitch, sizeof(double)*parasize, maxIters, cudaMemcpyDeviceToHost);
	cudaErr = cudaFree(d_ColVec);

	/****************
	Modified inliers by the model which is defined by hypo-inliers, then estimated a new model by inlers.
	Furthermore, it should be checked whether the new model is enough better by the criteria: 
		1. enough inliers(such as inliers>0.9*pcsize)
		2. model error is small enough.
	By-product:
		1. Model error, 2. point to curve distances.
	*****************/
	double* dev_xs, *dev_ys;
	cudaErr = cudaMalloc((void**)&dev_xs, sizeof(double)*pcsize);
	cudaErr = cudaMalloc((void**)&dev_ys, sizeof(double)*pcsize);
	cudaErr = cudaMemcpy(dev_xs, xvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(dev_ys, yvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);

	/***********
	Step1: Check out all inliers for the fitted model by using hypo-inliers in the proceding process.
	************/

	int* d_inlierNum, *InlierNum;
	bool* d_bInOut, *bInOut;
	double* d_ModelErr, *d_Dists;
	size_t bInOutPitch, distPitch;
	InlierNum = (int*)malloc(sizeof(int)*maxIters);
	bInOut = (bool*)malloc(sizeof(bool)*pcsize*maxIters);
	cudaErr = cudaMalloc((void**)&d_inlierNum, sizeof(int)*maxIters);
	cudaErr = cudaMemset(d_inlierNum, 0, sizeof(int)*maxIters);
	cudaErr = cudaMalloc((void**)&d_ModelErr, sizeof(double)*maxIters);
	cudaErr = cudaMemset(d_ModelErr, 0, sizeof(double)*maxIters);
	cudaErr = cudaMallocPitch((void**)&d_bInOut,&bInOutPitch, sizeof(bool)*pcsize, maxIters);
	cudaErr = cudaMallocPitch((void**)&d_Dists, &distPitch, sizeof(double)*pcsize, maxIters);

	//Get renewed inliers and estimate new model.
	//__global__ void CheckPointInOrOut(double* xs, double* ys, size_t pcsize, int maxIt, int parasize, double uTh, double lTh, size_t paraPitch, double* paras,
	//	int* inlierNum, size_t bInOutPitch, bool* bInOut, double* modelErr, size_t distPitch, double* dists)
	CheckPointInOrOut <<<(maxIters + 255) / 256, 256 >>> (dev_xs, dev_ys, pcsize, maxIters, parasize, uTh, lTh,
		parasPitch, d_Paras, d_inlierNum, bInOutPitch, d_bInOut, d_ModelErr, distPitch, d_Dists);
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();
	cudaErr = cudaMemcpy(InlierNum, d_inlierNum, sizeof(int)*maxIters, cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy2D(bInOut, sizeof(bool)*pcsize, d_bInOut, bInOutPitch, sizeof(bool)*pcsize, maxIters, cudaMemcpyDeviceToHost);

	//**********Free spaces
	cudaErr = cudaFree(dev_hypox);
	dev_hypox = NULL;
	cudaErr = cudaFree(dev_hypoy);
	dev_hypoy = NULL;
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

	//2. Resign new spaces:
	/**********
	IMPORTANT: In the part, there are TWO size will be modified,
	(1) The iteration number, from (num) to (variableNum);
	(2) The number of inlier points, for each iteration, this size will be various.
	***********/
	int variableNum = 0, curID = 0;
	for (int i = 0; i < num; i++)
	{
		if (InlierNum[i] > cols)
		{
			variableNum++;
		}
	}
	
	int *ModifyInlierNum, maxInliers=0;
	ModifyInlierNum = (int*)malloc(sizeof(int)*variableNum);
	bool *ModifybInOut;
	ModifybInOut = (bool*)malloc(sizeof(bool*)*variableNum*pcsize);
	
	int curIt = 0;
	for (int i = 0; i < num; i++)
	{
		if (InlierNum[i] > parasize)
		{
			ModifyInlierNum[curIt] = InlierNum[i];
			if (InlierNum[i] > maxInliers)
			{
				maxInliers = InlierNum[i];
			}
			/*for (size_t jj = 0; jj < pcsize; jj++)
			{
				ModifybInOut[curIt*pcsize + jj] = bInOut[i*pcsize + jj];
			}*/
			cudaErr = cudaMemcpy(ModifybInOut+curIt*pcsize, bInOut+i*pcsize, sizeof(bool)*pcsize, cudaMemcpyHostToHost);
			curIt++;
		}
	}

	/***********
	Step2: Setup matrices of As.
	************/
	num = variableNum;
	rows = maxInliers;
	cudaErr = cudaMalloc((void**)&d_inlierNum, sizeof(int)*num);
	cudaErr = cudaMemcpy(d_inlierNum, ModifyInlierNum, sizeof(int)*num, cudaMemcpyHostToDevice);
	cudaErr = cudaMallocPitch((void**)&d_bInOut, &bInOutPitch, sizeof(bool)*pcsize, num);
	cudaErr = cudaMemcpy2D(d_bInOut, bInOutPitch, ModifybInOut, sizeof(bool)*pcsize, sizeof(bool)*pcsize, num, cudaMemcpyHostToDevice);
	Aextent = make_cudaExtent(sizeof(double)*rows, cols, num);
	cudaErr = cudaMalloc3D(&d_Amat, Aextent);
	//__global__ void SetupMatriceswithVariousSizes(double* xvals, int maxIt, int pcSize, int paraSize,
	//	int* inlierNum, size_t bInOutPitch, bool*bInOrOut, cudaExtent Aextent, cudaPitchedPtr Amat)
	SetupMatriceswithVariousSizes <<<(maxIters + 255) / 256, 256 >>> (dev_xs, num, pcsize, parasize, d_inlierNum,
		bInOutPitch, d_bInOut, Aextent, d_Amat);
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaGetLastError();

	/***********
	Step3: Get parameters results from As.
	************/
	//Some spaces for VariateAs
	double *d_CurYs;
	size_t curYsPitch;
	cudaErr = cudaMallocPitch((void**)&d_ColVec, &colVecPitch, sizeof(double)*rows, num);
	cudaErr = cudaMallocPitch((void**)&d_CurYs, &curYsPitch, sizeof(double*)*rows, num);
	cudaErr = cudaMallocPitch((void**)&d_Paras, &parasPitch, sizeof(double)*cols, num);
	//__global__ void GetModelParawithVariateAs(double* yvals, size_t pcsize,int maxIt,int paraSize, cudaExtent Aextent, cudaPitchedPtr Amat, int* inlierNum,
	//size_t colVPitch, double* colVecs, size_t allYsPitch, double* allYs, size_t bInOutPitch, bool* bInOut, size_t paraPitch, double* paras)
	GetModelParawithVariateAs <<<(maxIters + 255) / 256, 256 >>> (dev_ys, pcsize, num, parasize, Aextent, d_Amat,
		d_inlierNum, colVecPitch, d_ColVec, curYsPitch, d_CurYs, bInOutPitch, d_bInOut, parasPitch, d_Paras);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	//Free temparory spaces:
	cudaErr = cudaFree(d_ColVec);
	d_ColVec = NULL;
	cudaErr = cudaFree(d_CurYs);
	d_CurYs = NULL;

	cudaErr = cudaMalloc((void**)&d_ModelErr, sizeof(double)*num);
	cudaErr = cudaMallocPitch((void**)&d_Dists, &distPitch, sizeof(double)*pcsize, num);
	//__global__ void GetModelSqDist(double* xs, double* ys,size_t pcsize, int maxIt, int parasize, size_t paraPitch, double* paras, int* inlierNum,
	//double* modelErr, size_t distPitch, double* dists)
	GetModelSqDist <<<(maxIters + 255) / 256, 256 >>> (dev_xs, dev_ys, pcsize, num, parasize, parasPitch, d_Paras,
		d_inlierNum, d_ModelErr, distPitch, d_Dists);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaMemcpy(modelErr, d_ModelErr, sizeof(double)*num, cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy2D(dists, sizeof(double)*pcsize, d_Dists, distPitch, sizeof(double)*pcsize, num, cudaMemcpyDeviceToHost);

	//***********Free spaces***********
	//Host:
	free(InlierNum);
	InlierNum = NULL;

	free(bInOut);
	bInOut = NULL;

	free(hst_hypoIDs);
	hst_hypoIDs = NULL;

	free(h_Paras);
	h_Paras = NULL;

	free(ModifyInlierNum);
	ModifyInlierNum = NULL;
	
	//Device space:
	cudaErr = cudaFree(dev_hypox);
	dev_hypox = NULL;
	cudaErr = cudaFree(dev_hypoy);
	dev_hypoy = NULL;

	cudaErr = cudaFree(d_Amat.ptr);
	d_Amat.ptr = NULL;

	cudaErr = cudaFree(dev_xs);
	dev_xs = NULL;
	cudaErr = cudaFree(dev_ys);
	dev_ys = NULL;

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