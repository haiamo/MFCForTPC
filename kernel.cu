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

__global__ void SetupMatriceswithVariousSizes(double* xvals, int maxIt, size_t pcsize, int paras, int* inlierNum, bool**bInOrOut,  double** Amat)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		int curInliers = *(inlierNum + tid);
		bool* curInOrOut = *(bInOrOut + tid);
		double* curA = *(Amat + tid);
		int Arowi = 0;
		if (curInliers > paras)
		{
			for (size_t Ptrowi = 0; Ptrowi < pcsize; Ptrowi++)
			{
				if (curInOrOut[Ptrowi])
				{
					curA[Arowi] = 1;
					for (int Acoli = 1; Acoli < paras; Acoli++)
					{
						curA[Acoli*curInliers + Arowi] = curA[(Acoli - 1)*curInliers + Arowi] * xvals[Ptrowi];
					}
					Arowi++;
				}
			}
		}
	}
}

__global__ void GetModelPara(double** QRs, double** taus, double* hypoy, int maxIt,int inliers/*rows*/, int parasize/*columns*/,
	double** tmpVs, double** tmpHs, double** tmpQs, double** Q,double** R, double** paras)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	int yBegId = tid*inliers;
	if (tid < maxIt)
	{
		int rows = inliers, cols = parasize, curID = 0;
		double* curQR = *(QRs + tid), *curTau = *(taus + tid),
			*outR=*(R+tid), *outQ=*(Q+tid), *outPara=*(paras+tid);
		//R in the upper triangle matrix, and candidate Q is lower. Matrix in column-major and stored in line.
		//Q should be an Identity matrix.
		
		/*extern __shared__ double sharedSpace[];
		//double* outR = sharedSpace;//length:rows*cols
		double* tmpv = sharedSpace;//length:rows
		double* tmpH = (double*)&tmpv[rows];//length:rows*rows
		double* tmpQ = (double*)&tmpH[rows*rows];//length:rows*rows*/
		double *tmpv = *(tmpVs + tid), *tmpH = *(tmpHs + tid), *tmpQ = *(tmpQs + tid);
		double tmpVal = 0.0;
		for (int ii = 0; ii < cols; ii++)
		{
			for (int jj = 0; jj < rows; jj++)
			{
				curID = rows*ii + jj;
				tmpVal = *(curQR + curID);
				//R:
				if (jj <= ii)
				{
					outR[curID] = tmpVal;
				}
				else
				{
					outR[curID] = 0.0;
				}
			}
			outPara[ii] = 0.0;
		}

		//double* resLine = new double[cols];// = (double*)malloc(sizeof(double)*cols);
		//memset(resLine, 0.0, sizeof(double)*cols);
		for (int ii = 0; ii < cols; ii++)
		{
			//v:
			for (int jj = 0; jj < rows; jj++)
			{
				curID = rows*ii + jj;
				tmpVal = *(curQR + curID);
				
				if (jj < ii)
				{
					tmpv[jj] = 0.0;
				}
				else if (jj == ii)
				{
					tmpv[jj] = 1.0;
				}
				else if (jj > ii)
				{
					tmpv[jj] = tmpVal;
				}
			}
			__syncthreads();

			//H=I-tau*v*v^t, where H, I and v have rows*rows, rows*rows and rows*1 dimensions respectively.
			for (int kk = 0; kk < rows; kk++)
			{
				for (int mm = 0; mm < rows; mm++)
				{
					curID = mm + kk*rows;
					if (kk == mm)
					{
						tmpH[curID] = 1.0 - curTau[ii] * tmpv[mm] * tmpv[kk];
					}
					else
					{
						tmpH[curID] = -1.0 * curTau[ii] * tmpv[mm] * tmpv[kk];
					}

				}
			}
			__syncthreads();

			//Q=H1*H2*...Hk, where k=max(1,min(rows,cols)), where Q and Hs have rows*rows dimension.
			
			//Copy outQ(mm) to H(mm)
			for (int kk = 0; kk < rows; kk++)
			{
				for (int mm = 0; mm < rows; mm++)
				{
					tmpQ[kk*rows + mm] = outQ[kk*rows + mm];
				}
			}
			__syncthreads();
			
			//outQ(mm)=H(mm-1)*H(mm), where mm=0,1,2,...,k, H0=I
			for (int kk = 0; kk < rows; kk++)//Column of Q
			{
				for (int mm = 0; mm < rows; mm++)//Row of Q
				{
					tmpVal = 0.0;
					for (int tt = 0; tt < rows; tt++)
					{
						tmpVal += tmpQ[tt*rows + mm] * tmpH[kk*rows + tt];
					}
					outQ[kk*rows + mm] = tmpVal;
				}
			}
			__syncthreads();
		}

		//Solve linear system R1*alpha=Q1^T*y, where R=[R1 O]^T, Q=[Q1 Q2]^T, and Q is orthonomal matrix.
		// R1 has cols*cols size, Q1 has rows*cols size and y rows*1.
		for (int ii = cols - 1; ii >= 0; ii--)
		{
			//Calculate the ii-th value in Q1^T*y
			tmpVal = 0.0;
			for (int rowi = 0; rowi < rows; rowi++)
			{
				curID = ii*rows + rowi;
				tmpVal += outQ[curID] * hypoy[yBegId+rowi];
			}

			//Calculate the ii-th parameter in alpha
			for (int coli = ii+1; coli < cols; coli++)
			{
				curID = coli*rows + ii;
				tmpVal -= outPara[coli] * outR[curID];
			}
			outPara[ii] = tmpVal / outR[ii*rows + ii];
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

__global__ void GetModelSqDist(double* xs, double* ys,size_t pcsize, int maxIt, int parasize, double** paras, int* inlierNum,
							 double* modelErr, double** dists)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		int Arows = *(inlierNum + tid), Acols = parasize;
		if (Arows > Acols)
		{
			double* curPara = *(paras + tid), *outErr = modelErr + tid, *outDist = *(dists + tid);
			if (NULL != curPara)
			{
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
}

__global__ void CheckPointInOrOut(double* xs, double* ys, size_t pcsize, int maxIt, int parasize,double uTh,double lTh, double** paras,
	int* inlierNum, bool** bInOut, double* modelErr, double** dists)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		double* curPara = *(paras + tid), *outErr = modelErr + tid, *outDist = *(dists + tid);
		*outErr = 0.0;
		bool *curbInOut = *(bInOut + tid);
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

__global__ void GetInliers(size_t pcsize, int maxIt, int* inlierNum, bool** bInOut, int** inlierIDs)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		int inNum = *(inlierNum + tid), curInID = 0;
		int* curIDs = *(inlierIDs + tid);
		bool* curbInOut = *(bInOut + tid);
		for (size_t ii = 0; ii < pcsize; ii++)
		{
			if (curbInOut[ii] && curInID<inNum)
			{
				curIDs[curInID] = ii;
				curInID++;
			}
		}
	}
}


__global__ void GetModelParawithVariateAs(double* yvals, size_t pcsize,int maxIt,int paraSize, double** Amat, int* inlierNum, 
	   double** colVecs, double** allYs, bool** bInlier, double** paras)
{
	//This function uses QR decomposition to get the least-square model for each iteration. The size of input matrices A can be various.
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		int Arows = *(inlierNum + tid), Acols=paraSize;
		if (Arows > Acols)
		{
			bool* curbInlier = *(bInlier + tid);
			double* curA = *(Amat + tid);
			double *outPara = *(paras + tid), tmpVal = 0.0;

			int curID = 0;
			double vnorm = 0.0, sigma = 0.0, beta = 0.0, ytv = 0.0;
			double* col_vec = *(colVecs+tid), *Atv_vec = (double*)malloc(sizeof(double)*Acols), *curYs = *(allYs+tid);
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
					curID = coli*Arows + rowi;
					vnorm += curA[curID] * curA[curID];
					col_vec[rowi] = curA[curID];
				}
				vnorm = sqrt(vnorm);

				//Compute beta
				if (curA[coli*Arows + coli] < 0)
				{
					sigma = -vnorm;
				}
				else
				{
					sigma = vnorm;
				}
				col_vec[coli] +=sigma;
				beta = 1 / sigma / (sigma + curA[coli*Arows+coli]);

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
						curID = colj*Arows + rowj;
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
						curID = colj*Arows + rowj;
						curA[curID] -= beta*col_vec[rowj] * Atv_vec[colj];
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
					curID = coli*Arows + rowi;
					tmpVal -= outPara[coli] * curA[curID];
				}
				outPara[rowi] = tmpVal / curA[rowi*Arows + rowi];
			}
			free(Atv_vec);
			Atv_vec = NULL;
		}
	}
}

extern "C" cudaError_t RANSACOnGPU(double* xvals, double* yvals, size_t pcsize, int maxIters, int minInliers, int parasize, double uTh, double lTh,
	double* &hst_hypox, double* &hst_hypoy, double* &As, double** &Qs, double** &taus, double** &Rs, double**& paras,
	double* &modelErr, double** &dists, int &hypoIters)
{
	//Prepare device space and parameters.
	cudaError_t cudaErr;
	cublasStatus_t stat;
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
	if (cudaError_t::cudaSuccess != cudaErr)
	{
		return cudaErr;
	}
	cudaErr = cudaMalloc((void**)&dev_hypoy, sizeof(double)*maxIters*minInliers);
	if (cudaError_t::cudaSuccess != cudaErr)
	{
		return cudaErr;
	}

	cudaErr = cudaMemcpy(dev_hypox, hst_hypox, sizeof(double)*maxIters*minInliers, cudaMemcpyHostToDevice);
	if (cudaError_t::cudaSuccess != cudaErr)
	{
		return cudaErr;
	}
	cudaErr = cudaMemcpy(dev_hypoy, hst_hypoy, sizeof(double)*maxIters*minInliers, cudaMemcpyHostToDevice);
	if (cudaError_t::cudaSuccess != cudaErr)
	{
		return cudaErr;
	}

	int rows = minInliers, cols = parasize;
	int num = maxIters, ltau = cols;//Number of matrices, known as the maximum iterations of RANSAC
	double *h_Amat=As;// *h_Qmat, *h_tauvec;
	cudaPitchedPtr d_Amat;
	cudaExtent Aextent;
	cudaMemcpy3DParms AcpParm = { 0 };
	Aextent = make_cudaExtent(sizeof(double)*rows, cols, num);
	cudaErr = cudaMalloc3D(&d_Amat, Aextent);
	SetupMatrices <<<(maxIters + 255) / 256, 256 >>> (dev_hypox, maxIters, d_Amat, Aextent);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	AcpParm.dstPtr = make_cudaPitchedPtr((void*)h_Amat, sizeof(double)*rows, rows, cols);
	AcpParm.srcPtr = d_Amat;
	AcpParm.extent = Aextent;
	AcpParm.kind = cudaMemcpyDeviceToHost;
	cudaErr = cudaMemcpy3D(&AcpParm);

	//__global__ void GetHypoModelPara(double* yvals, int maxIt, size_t pcsize, int parasize, cudaPitchedPtr Amat, cudaExtent Aextent,
		//size_t colvPitch, double** colVecs, size_t paraPitch, double* paras)
	double* d_colVec, *d_paras, *h_paras;
	size_t colVecPitch, parasPitch;
	h_paras = (double*)malloc(sizeof(double)*maxIters*parasize);
	cudaErr = cudaMallocPitch((void**)&d_colVec, &colVecPitch, sizeof(double)*minInliers, maxIters);
	cudaErr = cudaMallocPitch((void**)&d_paras, &parasPitch, sizeof(double)*parasize, maxIters);
	cudaErr = cudaMemset2D(d_colVec, colVecPitch, 0, sizeof(double)*minInliers, maxIters);
	cudaErr = cudaMemset2D(d_paras, parasPitch, 0, sizeof(double)*parasize, maxIters);
	GetHypoModelPara <<<(maxIters + 255) / 256, 256 >>> (dev_hypoy, maxIters, minInliers, parasize, d_Amat, Aextent,
		colVecPitch, d_colVec, parasPitch, d_paras);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaMemcpy2D(h_paras, sizeof(double)*parasize, d_paras, parasPitch, sizeof(double)*parasize, maxIters, cudaMemcpyDeviceToHost);
	cudaErr = cudaFree(d_colVec);
	cudaErr = cudaFree(d_paras);
	
	/*********************
	Get the 2D curve model by Least Square Method with QR Decomposition
	Sove the equation A*a=y, where A has the size n*m, a has m*1 and y has n*1.
	In this problem, n(rows) is the size of hypo-inliers Points, m(cols) equals to the order of fitted ploynomial plus one(m=order+1).
	For parallel computing, there are MaxIteration(mi:num) matrices in total.
	All matrices is column-major stored continuously.
	**********************/

	double **Aarray, **Tauarray=taus, **Qarray=Qs, **Rarray=Rs, **Paraarray=paras;
	Aarray = (double**)malloc(num * sizeof(double*));

	//Set Qarray equal to Identity
	for (int iti = 0; iti < num; iti++)
	{
		memset(Qarray[iti], 0, sizeof(double)*rows*rows);
		for (int ii = 0; ii < rows; ii++)
		{
			(Qarray[iti])[ii*rows + ii] = 1.0;
		}
	}

	for (int i = 0; i < num; i++)
	{
		Aarray[i] = (double*)malloc(rows*cols * sizeof(double));
	}

	//Create host pointer array to device matrix storage
	double **d_Aarray, **d_Tauarray, **h_d_Aarray, **h_d_Tauarray;
	double **d_Qarray, **d_Rarray, **h_d_Qarray, **h_d_Rarray;
	double **d_Paraarray, **h_d_Paraarray;
	double **d_Distarray, **h_d_Distarray, *d_ModelErr;
	h_d_Aarray = (double**)malloc(num * sizeof(double*));
	h_d_Tauarray = (double**)malloc(num * sizeof(double*));
	h_d_Qarray = (double**)malloc(num * sizeof(double*));
	h_d_Rarray = (double**)malloc(num * sizeof(double*));
	h_d_Paraarray = (double**)malloc(num * sizeof(double*));
	h_d_Distarray = (double**)malloc(num * sizeof(double*));
	cudaErr = cudaMalloc((void**)&d_ModelErr, num * sizeof(double));

	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaMalloc((void**)&h_d_Aarray[i], rows*cols * sizeof(double));
		cudaErr = cudaMalloc((void**)&h_d_Tauarray[i], ltau * sizeof(double));
		cudaErr = cudaMalloc((void**)&h_d_Qarray[i], rows*rows * sizeof(double));
		cudaErr = cudaMemcpy(h_d_Qarray[i], Qarray[i], sizeof(double)*rows*rows, cudaMemcpyHostToDevice);
		cudaErr = cudaMalloc((void**)&h_d_Rarray[i], rows*cols * sizeof(double));
		cudaErr = cudaMalloc((void**)&h_d_Paraarray[i], cols * sizeof(double));
		cudaErr = cudaMalloc((void**)&h_d_Distarray[i], pcsize * sizeof(double));
	}

	//Copy the host array of device pointers to the device
	stat = cublasAlloc(num, sizeof(double*), (void**)&d_Aarray);
	stat = cublasAlloc(num, sizeof(double*), (void**)&d_Tauarray);
	cudaErr = cudaMalloc((void**)&d_Qarray, num * sizeof(double*));
	cudaErr = cudaMalloc((void**)&d_Rarray, num * sizeof(double*));
	cudaErr = cudaMalloc((void**)&d_Paraarray, num * sizeof(double*));
	cudaErr = cudaMalloc((void**)&d_Distarray, num * sizeof(double*));

	cudaErr = cudaMemcpy(d_Qarray, h_d_Qarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Rarray, h_d_Rarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Paraarray, h_d_Paraarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Distarray, h_d_Distarray, num * sizeof(double*), cudaMemcpyHostToDevice);

	//fill Aarray
	int index = 0;
	for (int k = 0; k < num; k++)
	{
		for (int j = 0; j < cols; j++)
		{
			for (int i = 0; i < rows; i++)
			{
				index = j*rows + i;
				(Aarray[k])[index] = h_Amat[k*rows*cols+index];
			}
		}
	}
	
	//Create cublas instance
	cublasHandle_t handle;
	
	stat = cublasCreate(&handle);
	
	//Set input matrices onto device
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaMemcpy(h_d_Aarray[i], Aarray[i], rows*cols * sizeof(double), cudaMemcpyHostToDevice);
		cudaErr = cudaMemcpy(h_d_Tauarray[i], Tauarray[i], ltau * sizeof(double), cudaMemcpyHostToDevice);
	}
	cudaErr = cudaMemcpy(d_Aarray, h_d_Aarray, num *sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Tauarray, h_d_Tauarray, num *sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaDeviceSynchronize();
	int *info, lda = rows;
	info = (int*)malloc(sizeof(int)*num);
	stat = cublasDgeqrfBatched(handle, rows, cols, d_Aarray, rows, d_Tauarray, info, num);
	cudaErr = cudaDeviceSynchronize();
	//Retrieve result matrix from device,just for checking result
	cudaErr = cudaMemcpy(h_d_Aarray, d_Aarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(h_d_Tauarray, d_Tauarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaMemcpy(Aarray[i], h_d_Aarray[i], rows*cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaErr = cudaMemcpy(Tauarray[i], h_d_Tauarray[i], ltau * sizeof(double), cudaMemcpyDeviceToHost);

		//Fill h_Amat for output, just for testing.
		for (int jj = 0; jj < cols; jj++)
		{
			for (int ii = 0; ii < rows; ii++)
			{
				index = i*rows*cols + jj*rows + ii;
				h_Amat[index] = (Aarray[i])[jj*rows + ii];
			}
		}
	}
	//Global spaces for GetModelPara
	double **h_d_tmpV, **h_d_tmpH, **h_d_tmpQ;
	double **d_tmpV, **d_tmpH, **d_tmpQ;
	h_d_tmpV = (double**)malloc(sizeof(double*)*num);
	h_d_tmpH = (double**)malloc(sizeof(double*)*num);
	h_d_tmpQ = (double**)malloc(sizeof(double*)*num);
	cudaErr = cudaMalloc((void**)&d_tmpV, sizeof(double*)*num);
	cudaErr = cudaMalloc((void**)&d_tmpH, sizeof(double*)*num);
	cudaErr = cudaMalloc((void**)&d_tmpQ, sizeof(double*)*num);
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaMalloc((void**)&h_d_tmpV[i], sizeof(double)*rows);
		cudaErr = cudaMalloc((void**)&h_d_tmpH[i], sizeof(double)*rows*rows);
		cudaErr = cudaMalloc((void**)&h_d_tmpQ[i], sizeof(double)*rows*rows);
	}
	cudaErr = cudaMemcpy(d_tmpV, h_d_tmpV, sizeof(double*)*num, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_tmpH, h_d_tmpH, sizeof(double*)*num, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_tmpQ, h_d_tmpQ, sizeof(double*)*num, cudaMemcpyHostToDevice);
	//__global__ void GetModelPara(double** QRs, double** taus, double* hypoy, int maxIt, int inliers/*rows*/, int parasize/*columns*/,
	//	double** tmpVs, double** tmpHs, double** tmpQs, double** Q, double** R, double** paras)
	GetModelPara <<<(maxIters + 255) / 256, 256>>>(d_Aarray, d_Tauarray, dev_hypoy, num,
		rows, cols,d_tmpV,d_tmpH,d_tmpQ, d_Qarray, d_Rarray, d_Paraarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	//Free tempory spaces:
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaFree(h_d_tmpV[i]);
		cudaErr = cudaFree(h_d_tmpH[i]);
		cudaErr = cudaFree(h_d_tmpQ[i]);
	}
	free(h_d_tmpV);
	h_d_tmpV = NULL;
	free(h_d_tmpH);
	h_d_tmpH = NULL;
	free(h_d_tmpQ);
	h_d_tmpQ = NULL;

	cudaErr = cudaFree(d_tmpV);
	d_tmpV = NULL;
	cudaErr = cudaFree(d_tmpH);
	d_tmpH = NULL;
	cudaErr = cudaFree(d_tmpQ);
	d_tmpQ = NULL;

	//Retrieve reslut from device, just for result checking.
	cudaErr = cudaMemcpy(h_d_Qarray, d_Qarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(h_d_Rarray, d_Rarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(h_d_Paraarray, d_Paraarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaDeviceSynchronize();
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaMemcpy(Qarray[i], h_d_Qarray[i], rows*rows * sizeof(double), cudaMemcpyDeviceToHost);
		cudaErr = cudaMemcpy(Rarray[i], h_d_Rarray[i], rows*cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaErr = cudaMemcpy(Paraarray[i], h_d_Paraarray[i], cols * sizeof(double), cudaMemcpyDeviceToHost);
	}
	cudaErr = cudaDeviceSynchronize();

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

	int* d_inlierNum, *InlierNum;
	bool** d_bInOut, **h_d_bInOut, **bInOut;
	//int** d_InlierID, **h_d_InlierID, **InlierID;
	//h_d_Inliers = (int**)malloc(sizeof(int*)*num);
	h_d_bInOut = (bool**)malloc(sizeof(bool*)*num);
	//h_d_InlierID = (int**)malloc(sizeof(int*)*num);

	cudaErr = cudaMalloc((void**)&d_inlierNum, sizeof(int)*num);
	//cudaErr = cudaMalloc((void**)&d_Inliers, sizeof(int*)*num);
	cudaErr = cudaMalloc((void**)&d_bInOut, sizeof(bool*)*num);
	bInOut = (bool**)malloc(sizeof(bool*)*num);
	InlierNum = (int*)malloc(sizeof(int)*num);
	//cudaErr = cudaMalloc((void**)&d_InlierID, sizeof(int*)*num);
	//InlierID = (int**)malloc(sizeof(int*)*num);
	rows = pcsize;
	for (int ii = 0; ii < num; ii++)
	{
		cudaErr = cudaMalloc((void**)&h_d_bInOut[ii], sizeof(bool)*rows);
		//InlierID[ii] = (int*)malloc(sizeof(int)*rows);
		bInOut[ii] = (bool*)malloc(sizeof(bool)*rows);
	}
	cudaErr = cudaMemcpy(d_bInOut, h_d_bInOut, sizeof(bool*)*num, cudaMemcpyHostToDevice);

	//Get renewed inliers and estimate new model.
	CheckPointInOrOut <<<(maxIters+255)/256,256>>>(dev_xs, dev_ys, pcsize, maxIters, parasize, uTh, lTh, d_Paraarray,
		d_inlierNum, d_bInOut, d_ModelErr, d_Distarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaMemcpy(InlierNum, d_inlierNum, sizeof(int)*num, cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(modelErr, d_ModelErr, num * sizeof(double), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(h_d_bInOut, d_bInOut, sizeof(bool*)*num, cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(h_d_Distarray, d_Distarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	for (int ii = 0; ii < num; ii++)
	{
		//rows = InlierNum[ii];
		//cudaErr = cudaMalloc((void**)&h_d_Inliers[ii], sizeof(int)*rows);
		//cudaErr = cudaMalloc((void**)&h_d_InlierID[ii], sizeof(double)*rows);
		cudaErr = cudaMemcpy(bInOut[ii], h_d_bInOut[ii], sizeof(bool)*rows,cudaMemcpyDeviceToHost);
		cudaErr = cudaMemcpy(dists[ii], h_d_Distarray[ii], pcsize * sizeof(double), cudaMemcpyDeviceToHost);
	}
	hypoIters = num;

	//cudaErr = cudaMemcpy(d_Inliers, h_d_Inliers, sizeof(int*)*num, cudaMemcpyHostToDevice);
	/*cudaErr = cudaMemcpy(d_InlierID, h_d_InlierID, sizeof(int*)*num, cudaMemcpyHostToDevice);

	GetInliers<<<(maxIters+255)/256,256>>>(pcsize, maxIters, d_inlierNum, d_bInOut, d_InlierID);
	cudaErr = cudaMemcpy(h_d_InlierID, d_InlierID, sizeof(int*)*num, cudaMemcpyDeviceToHost);
	for (int ii = 0; ii < num; ii++)
	{
		cudaErr = cudaMemcpy(InlierID[ii], h_d_InlierID[ii], sizeof(int*)*rows, cudaMemcpyDeviceToHost);
	}*/

	//**********Resize spaces
	//1. Free spaces
	//Host-device pointer:
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaFree(h_d_Aarray[i]);
		cudaErr = cudaFree(h_d_Tauarray[i]);
		cudaErr = cudaFree(h_d_Qarray[i]);
		cudaErr = cudaFree(h_d_Rarray[i]);
		cudaErr = cudaFree(h_d_Distarray[i]);
		cudaErr = cudaFree(h_d_bInOut[i]);
	}

	free(Aarray);
	Aarray = NULL;

	free(h_d_Aarray);
	free(h_d_Tauarray);
	free(h_d_Qarray);
	free(h_d_Rarray);
	free(h_d_Distarray);
	free(h_d_bInOut);
	h_d_Aarray = NULL;
	h_d_Tauarray = NULL;
	h_d_Qarray = NULL;
	h_d_Rarray = NULL;
	h_d_Distarray = NULL;
	h_d_bInOut = NULL;

	cudaErr = cudaFree(dev_hypox);
	dev_hypox = NULL;
	cudaErr = cudaFree(dev_hypoy);
	dev_hypoy = NULL;

	cudaErr = cudaFree(d_Aarray);
	d_Aarray = NULL;
	cudaErr = cudaFree(d_Tauarray);
	d_Tauarray = NULL;
	cudaErr = cudaFree(d_Qarray);
	d_Qarray = NULL;
	cudaErr = cudaFree(d_Rarray);
	d_Rarray = NULL;
	cudaErr = cudaFree(d_ModelErr);
	d_ModelErr = NULL;
	cudaErr = cudaFree(d_Distarray);
	d_Distarray = NULL;
	cudaErr = cudaFree(d_bInOut);
	d_bInOut = NULL;

	cudaErr = cudaFree(d_inlierNum);
	d_inlierNum = NULL;
	//**********Free Spaces

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
	
	int *ModifyInlierNum;
	ModifyInlierNum = (int*)malloc(sizeof(int)*num);
	bool **ModifybInOut;
	ModifybInOut = (bool**)malloc(sizeof(bool*)*variableNum);
	
	for (int i = 0; i < num; i++)
	{
		if (InlierNum[i] > cols)
		{
			ModifyInlierNum[curID] = InlierNum[i];
			ModifybInOut[curID] = bInOut[i];
			curID++;
		}
	}

	num = variableNum;
	cudaErr = cudaMalloc((void**)&d_inlierNum, sizeof(int)*num);
	cudaErr = cudaMemcpy(d_inlierNum, ModifyInlierNum, sizeof(int)*num, cudaMemcpyHostToDevice);
	cudaErr = cudaMalloc((void**)&d_ModelErr, num * sizeof(double));
	cudaErr = cudaMemset(d_ModelErr, 0, sizeof(double)*num);
	memset(modelErr, 0, sizeof(double)*maxIters);

	Aarray = (double**)malloc(num * sizeof(double*));
	h_d_Aarray = (double**)malloc(num * sizeof(double*));
	h_d_Tauarray = (double**)malloc(num * sizeof(double*));
	h_d_Qarray = (double**)malloc(num * sizeof(double*));
	h_d_Rarray = (double**)malloc(num * sizeof(double*));
	h_d_Paraarray = (double**)malloc(num * sizeof(double*));
	h_d_bInOut = (bool**)malloc(num * sizeof(bool*));
	cudaErr = cudaMalloc((void**)&d_bInOut, sizeof(bool*)*num);
	h_d_Distarray = (double**)malloc(num * sizeof(double*));
	cudaErr = cudaMalloc((void**)&d_Distarray, sizeof(double*)*num);

	for (int i = 0; i < num; i++)
	{
		rows = ModifyInlierNum[i];
		if (rows > 0)
		{
			cudaErr = cudaMalloc((void**)&h_d_Aarray[i], rows*cols * sizeof(double));
			cudaErr = cudaMalloc((void**)&h_d_Qarray[i], rows*rows * sizeof(double));
			cudaErr = cudaMalloc((void**)&h_d_Rarray[i], rows*cols * sizeof(double));
			Aarray[i] = (double*)malloc(rows*cols * sizeof(double));
			cudaErr = cudaMalloc((void**)&h_d_Tauarray[i], ltau * sizeof(double));
			cudaErr = cudaMalloc((void**)&h_d_Paraarray[i], cols * sizeof(double));
			cudaErr = cudaMalloc((void**)&h_d_bInOut[i], sizeof(bool*)*pcsize);
			cudaErr = cudaMemcpy(h_d_bInOut[i], ModifybInOut[i], sizeof(bool)*pcsize, cudaMemcpyHostToDevice);
			cudaErr = cudaMalloc((void**)&h_d_Distarray[i], rows * sizeof(double));
		}
	}

	//Copy the host array of device pointers to the device
	stat = cublasAlloc(num, sizeof(double*), (void**)&d_Aarray);
	stat = cublasAlloc(num, sizeof(double*), (void**)&d_Tauarray);
	cudaErr = cudaMalloc((void**)&d_Qarray, num * sizeof(double*));
	cudaErr = cudaMalloc((void**)&d_Rarray, num * sizeof(double*));
	cudaErr = cudaMalloc((void**)&d_Paraarray, num * sizeof(double*));

	cudaErr = cudaMemcpy(d_Aarray, h_d_Aarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Qarray, h_d_Qarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Rarray, h_d_Rarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Paraarray, h_d_Paraarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_bInOut, h_d_bInOut, num * sizeof(bool*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Distarray, h_d_Distarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	
	SetupMatriceswithVariousSizes <<<(maxIters+255)/256,256>>>(dev_xs, num, pcsize, parasize, d_inlierNum, d_bInOut, d_Aarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaMemcpy(h_d_Aarray, d_Aarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	for (int i = 0; i < num; i++)
	{
		rows = ModifyInlierNum[i];
		if (rows > 0)
		{
			cudaErr = cudaMemcpy(Aarray[i], h_d_Aarray[i], rows*cols * sizeof(double), cudaMemcpyDeviceToHost);
		}

	}

	//Some spaces for VariateAs
	double** h_d_ColVec, **d_ColVec, **h_d_CurYs, **d_CurYs;
	h_d_ColVec = (double**)malloc(sizeof(double*)*num);
	h_d_CurYs = (double**)malloc(sizeof(double*)*num);
	cudaErr = cudaMalloc((void**)&d_ColVec, sizeof(double*)*num);
	cudaErr = cudaMalloc((void**)&d_CurYs, sizeof(double*)*num);
	for (int i = 0; i < num; i++)
	{
		rows = ModifyInlierNum[i];
		if (rows > 0)
		{
			cudaErr = cudaMalloc((void**)&h_d_ColVec[i], sizeof(double)*rows);
			cudaErr = cudaMalloc((void**)&h_d_CurYs[i], sizeof(double)*rows);
		}
	}
	cudaErr = cudaMemcpy(d_ColVec, h_d_ColVec, sizeof(double*)*num, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_CurYs, h_d_CurYs, sizeof(double*)*num, cudaMemcpyHostToDevice);

	//__global__ void GetModelParawithVariateAs(double* yvals, size_t pcsize,int maxIt,int paraSize, double** Amat, int* inlierNum, 
	//double** colVecs, double** allYs, bool** bInlier, double** paras)
	GetModelParawithVariateAs<<<(maxIters+255)/256,256>>>(dev_ys, pcsize, num, parasize, d_Aarray,d_inlierNum,
		d_ColVec,d_CurYs,d_bInOut, d_Paraarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	//Free temparory spaces:
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaFree(h_d_ColVec[i]);
		cudaErr = cudaFree(h_d_CurYs[i]);
	}
	free(h_d_ColVec);
	h_d_ColVec = NULL;
	free(h_d_CurYs);
	h_d_CurYs = NULL;
	cudaErr = cudaFree(d_ColVec);
	d_ColVec = NULL;
	cudaErr = cudaFree(d_CurYs);
	d_CurYs = NULL;

	cudaErr = cudaMemcpy(h_d_Paraarray, d_Paraarray, sizeof(double*)*num, cudaMemcpyDeviceToHost);
	for (int i = 0; i < num; i++)
	{
		rows = ModifyInlierNum[i];
		if (rows > 0)
		{
			cudaErr = cudaMemcpy(Paraarray[i], h_d_Paraarray[i], cols * sizeof(double), cudaMemcpyDeviceToHost);
			if (cudaError_t::cudaSuccess != cudaErr)
			{
				return cudaErr;
			}
		}
	}

	//__global__ void GetModelSqDist(double* xs, double* ys, size_t pcsize, int maxIt, int parasize, double** paras, int* inlierNum,
	//	double* modelErr, double** dists)
	GetModelSqDist <<<(maxIters + 255) / 256, 256 >>> (dev_xs, dev_ys, pcsize, num, parasize, d_Paraarray, d_inlierNum, d_ModelErr, d_Distarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaMemcpy(h_d_Distarray, d_Distarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(modelErr, d_ModelErr, num * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < num; i++)
	{
		rows = ModifyInlierNum[i];
		if (rows > 0)
		{
			cudaErr = cudaMemcpy(dists[i], h_d_Distarray[i], rows * sizeof(double), cudaMemcpyDeviceToHost);
		}
		
	}

	//***********Free spaces***********
	//Host-device pointer:
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaFree(h_d_Aarray[i]);
		cudaErr = cudaFree(h_d_Tauarray[i]);
		cudaErr = cudaFree(h_d_Qarray[i]);
		cudaErr = cudaFree(h_d_Rarray[i]);
		cudaErr = cudaFree(h_d_Distarray[i]);

		//cudaErr = cudaFree(h_d_InlierID[i]);
		cudaErr = cudaFree(h_d_bInOut[i]);
	}

	//Host:
	free(hst_hypoIDs);
	hst_hypoIDs = NULL;

	free(Aarray);
	Aarray = NULL;

	free(h_paras);
	h_paras = NULL;

	free(h_d_Aarray);
	free(h_d_Tauarray);
	free(h_d_Qarray);
	free(h_d_Rarray);
	free(h_d_Distarray);
	h_d_Aarray = NULL;
	h_d_Tauarray = NULL;
	h_d_Qarray = NULL;
	h_d_Rarray = NULL;
	h_d_Distarray = NULL;

	//free(h_d_InlierID);
	//h_d_InlierID = NULL;
	free(h_d_bInOut);
	h_d_bInOut = NULL;

	free(ModifyInlierNum);
	ModifyInlierNum = NULL;
	
	//Device space:
	cudaErr = cudaFree(dev_hypox);
	dev_hypox = NULL;
	cudaErr = cudaFree(dev_hypoy);
	dev_hypoy = NULL;

	cudaErr = cudaFree(d_Amat.ptr);
	d_Amat.ptr = NULL;

	cudaErr = cudaFree(d_Aarray);
	d_Aarray = NULL;
	cudaErr = cudaFree(d_Tauarray);
	d_Tauarray = NULL;
	cudaErr = cudaFree(d_Qarray);
	d_Qarray = NULL;
	cudaErr = cudaFree(d_Rarray);
	d_Rarray = NULL;
	cudaErr = cudaFree(d_Distarray);
	d_Distarray = NULL;
	cudaErr = cudaFree(d_ModelErr);
	d_ModelErr = NULL;

	cudaErr = cudaFree(dev_xs);
	dev_xs = NULL;
	cudaErr = cudaFree(dev_ys);
	dev_ys = NULL;

	cudaErr = cudaFree(d_inlierNum);
	d_inlierNum = NULL;
	return cudaErr;
}