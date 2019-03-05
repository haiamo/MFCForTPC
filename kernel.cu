#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "cuda.h"
#include "device_functions.h"
#include "cublas.h"
#include "cublas_v2.h"
#include "cudaMain.h"
#include "stdio.h"

__global__ void SetupMatrices(double* xvals, double* yvals,int maxIt,int inliers,int paras, double* Amat)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	int begID = tid*inliers*paras;
	if (tid < maxIt)
	{
		for (int coli = 0; coli < paras; coli++)
		{
			for (int rowi = 0; rowi < inliers; rowi++)
			{
				if (coli == 0)
				{
					Amat[begID + rowi] = 1;
				}
				else
				{
					Amat[begID + coli*inliers + rowi] = Amat[begID + (coli - 1)*inliers + rowi] * xvals[tid*inliers + rowi];
				}
			}
		}
	}
}

__global__ void SetupMatricesOnDevice(double* xvals, int maxIt, size_t pcsize, int paras, int* inlierNum, int**inlierIDs,  double** Amat)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		int curInliers = *(inlierNum + tid);
		int* curInlierID = *(inlierIDs + tid);
		int begID = tid*pcsize*paras;
		for (int coli = 0; coli < paras; coli++)
		{
			for (int rowi = 0; rowi < curInliers; rowi++)
			{
				if (coli == 0)
				{
					Amat[begID][rowi] = 1;
				}
				else
				{
					Amat[begID + coli*curInliers][rowi] = Amat[begID + (coli - 1)*curInliers][rowi] * xvals[curInlierID[rowi]];
				}
			}
		}
	}
}

__global__ void GetModelPara(double** QRs, double** taus, double* hypoy, int maxIt,
							int inliers/*rows*/, int parasize/*columns*/,double** Q,double** R, double** paras)
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
		double *tmpv, *tmpH, *tmpQ;
		tmpv = (double*)malloc(sizeof(double)*rows);
		tmpH = (double*)malloc(sizeof(double)*rows*rows);
		tmpQ = (double*)malloc(sizeof(double)*rows*rows);
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
		
		free(tmpv);
		free(tmpH);
		free(tmpQ);
		tmpv = NULL;
		tmpH = NULL;
		tmpQ = NULL;
	}
}

__global__ void GetModelSqDist(double* xs, double* ys,size_t pcsize, int maxIt, int parasize, double** paras,
							 double* modelErr, double** dists)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		double* curPara = *(paras + tid), *outErr = modelErr + tid, *outDist = *(dists + tid);
		double yHat = 0.0;//y^^=paras[k]*x^(k-1)+...paras[0], k=order of fitted polynomial
		double xPows = 0.0, tmpDist = 0.0;
		*modelErr = 0.0;
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
			*outErr += tmpDist*tmpDist;
		}
	}
}

__global__ void CheckPointInOrOut(double* xs, double* ys, size_t pcsize, int maxIt, int parasize,double uTh,double lTh, double** paras,
	int* inlierNum, bool** bInOut)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		double* curPara = *(paras + tid);
		bool *curbInOut = *(bInOut + tid);
		int *curInlNum = inlierNum + tid;
		double yHat = 0.0;//y^^=paras[k]*x^(k-1)+...paras[0], k=order of fitted polynomial
		double xPows = 0.0, tmpDist = 0.0;
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
			if ((tmpDist > 0 && tmpDist - uTh<0.0) || (tmpDist < 0 && tmpDist + lTh > 0.0))
			{
				(*curInlNum)++;
				curbInOut[ii] = true;
			}
		}
	}
}

__global__ void GetInliers(double* xs, double* ys, size_t pcsize, int maxIt, int* inlierNum, bool** bInOut, double** inlier_xs,double** inlier_ys)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		int inNum = *(inlierNum + tid), curInID = 0;
		double* curXs = *(inlier_xs + tid), *curYs = *(inlier_ys + tid);
		bool* curDist = *(bInOut + tid);
		for (size_t ii = 0; ii < pcsize; ii++)
		{
			if (curDist[ii] && curInID<inNum)
			{
				curXs[curInID] = xs[ii];
				curYs[curInID] = ys[ii];
				curInID++;
			}
		}
	}
}


__global__ void GetModelParawithVariateAs(double** Amat, size_t pcsize,int* inlierNum, int maxIt,int paraSize,
	 double** inlier_ys, double** paras)
{
	//This function uses QR decomposition to get the least-square model for each iteration. The size of input matrices A can be various.
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	if (tid < maxIt)
	{
		double* curA = *(Amat + tid), *curYs = *(inlier_ys + tid);
		double *outPara = *(paras + tid), tmpVal=0.0;
		int Arows = *(inlierNum + tid), Acols=paraSize;
		int curID = 0;
		double vnorm = 0.0, sigma = 0.0, beta = 0.0, ytv = 0.0;
		double* col_vec = (double*)malloc(sizeof(double)*Arows), *Atv_vec = (double*)malloc(sizeof(double)*Acols);
		//QR decomposition by Householder reflection.
		for (int coli = 0; coli < Acols; coli++)
		{
			outPara[coli] = 0.0;
			//Compute column vector(col_vec) norm
			vnorm = 0.0;
			for (int rowi = 0; rowi < Arows; rowi++)
			{
				curID = coli*Arows + rowi;
				vnorm += curA[curID] * curA[curID];
				col_vec[rowi] = curA[curID];
			}
			vnorm = sqrt(vnorm);

			//Compute beta
			if (curA[coli*Arows + coli] < 0)
			{
				col_vec[coli] -= vnorm;
				sigma = -vnorm;
			}
			else
			{
				col_vec[coli] += vnorm;
				sigma = vnorm;
			}
			beta = 1 / sigma / (sigma + col_vec[coli]);

			//Compute A^t*col_vec and y^t*col_vec
			for (int colj = 0; colj < Acols; colj++)
			{
				for (int rowi = 0; rowi < Arows; rowi++)
				{
					curID = colj*Arows + rowi;
					Atv_vec[rowi] += curA[curID] * col_vec[rowi];
					if (colj == 0)
					{
						ytv += curYs[rowi] * col_vec[rowi];
					}
				}
			}

			//H(k)A(k)=A(k-1)-beta(k-1)*col_vec(k-1)*Atv_vec^t(k-1)
			//y(k)=y(k-1)-bata(k-1)*col_vec(k-1)*ytv(k-1)
			for (int colj = 0; colj < Acols; colj++)
			{
				for (int rowj = 0; rowj < Arows; rowj++)
				{
					curID = colj*Arows + rowj;
					curA[curID] -= beta*col_vec[colj] * Atv_vec[rowj];
					if (colj == 0)
					{
						curYs[rowj] -= beta*col_vec[colj] * ytv;
					}
				}
			}
		}
		//Now, A->QA=R, y->Qy; Aalpha=y->Ralpha=Qy, the next step is alpha=R^(-1)Qy
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
		free(col_vec);
		col_vec = NULL;
		free(Atv_vec);
		Atv_vec = NULL;
	}
}

extern "C" cudaError_t RANSACOnGPU(double* xvals, double* yvals, size_t pcsize, int maxIters, int minInliers, int parasize, double uTh, double lTh,
	double* &hst_hypox, double* &hst_hypoy, double* &As, double** &Qs, double** &taus, double** &Rs, double**& paras,
	double* &modelErr, double** &dists)
{
	//Prepare device space and parameters.
	cudaError_t cudaErr;
	cublasStatus_t stat;
	double *dev_hypox=NULL, *dev_hypoy;
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
	double* d_Amat, *h_Amat=As;// *h_Qmat, *h_tauvec;
	cudaErr = cudaMalloc((void**)&d_Amat, sizeof(double)*num*rows*cols);
	if (cudaError_t::cudaSuccess != cudaErr)
	{
		return cudaErr;
	}
	SetupMatrices<<<(maxIters+255)/256,256>>>(dev_hypox, dev_hypoy, maxIters, minInliers, parasize, d_Amat);
	cudaErr = cudaGetLastError();
	cudaErr = cudaMemcpy(h_Amat, d_Amat, sizeof(double)*num*rows*cols, cudaMemcpyDeviceToHost);
	if (cudaError_t::cudaSuccess != cudaErr)
	{
		return cudaErr;
	}

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
		memset(Qarray[iti], 0.0, sizeof(double)*rows*rows);
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
	
	GetModelPara <<<(maxIters + 255) / 256, 256>>>(d_Aarray, d_Tauarray, dev_hypoy, num,
		rows, cols, d_Qarray, d_Rarray, d_Paraarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
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

	//Get model error and point to curve distances.
	double* dev_xs, *dev_ys;
	cudaErr = cudaMalloc((void**)&dev_xs, sizeof(double)*pcsize);
	cudaErr = cudaMalloc((void**)&dev_ys, sizeof(double)*pcsize);
	cudaErr = cudaMemcpy(dev_xs, xvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(dev_ys, yvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);

	GetModelSqDist <<<(maxIters + 255) / 256, 256 >>> (dev_xs, dev_ys, pcsize, maxIters, parasize, d_Paraarray, d_ModelErr, d_Distarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaMemcpy(h_d_Distarray, d_Distarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(modelErr, d_ModelErr, num * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaMemcpy(dists[i], h_d_Distarray[i], pcsize * sizeof(double), cudaMemcpyDeviceToHost);
	}
	/****************
	Modified inliers by the model which is defined by hypo-inliers, then estimated a new model by inlers.
	Furthermore, it should be checked whether the new model is enough better by the criteria: 
		1. enough inliers(such as inliers>0.9*pcsize)
		2. model error is small enough.
	*****************/
	int* d_inlierNum, **d_Inliers, *InlierNum;
	bool** d_bInOut;
	double** d_inlierx, **d_inliery;
	cudaErr = cudaMalloc((void**)&d_inlierNum, sizeof(int)*num);
	cudaErr = cudaMalloc((void**)&d_Inliers, sizeof(int*)*num);
	cudaErr = cudaMalloc((void**)&d_bInOut, sizeof(bool*)*num);
	InlierNum = (int*)malloc(sizeof(int)*num);
	cudaErr = cudaMalloc((void**)&d_inlierx, sizeof(double*)*num);
	cudaErr = cudaMalloc((void**)&d_inliery, sizeof(double*)*num);
	for (int ii = 0; ii < num; ii++)
	{
		cudaErr = cudaMalloc((void**)&d_Inliers[ii], sizeof(int)*rows);
		cudaErr = cudaMalloc((void**)&d_bInOut[ii], sizeof(bool)*rows);
		cudaErr = cudaMalloc((void**)&d_inlierx[ii], sizeof(double)*rows);
		cudaErr = cudaMalloc((void**)&d_inliery[ii], sizeof(double)*rows);
	}

	//Get renewed inliers and estimate new model.
	CheckPointInOrOut <<<(maxIters+255)/256,256>>>(dev_xs, dev_ys, pcsize, maxIters, parasize, uTh, lTh, d_Paraarray, d_inlierNum, d_bInOut);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaMemcpy(InlierNum, d_inlierNum, sizeof(int)*num, cudaMemcpyDeviceToHost);

	GetInliers<<<(maxIters+255)/256,256>>>(dev_xs, dev_ys, pcsize, maxIters, d_inlierNum, d_bInOut, d_inlierx, d_inliery);

	//Resize spaces
	//1. Free spaces
	//Host-device pointer:
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaFree(h_d_Aarray[i]);
		cudaErr = cudaFree(h_d_Tauarray[i]);
		cudaErr = cudaFree(h_d_Qarray[i]);
		cudaErr = cudaFree(h_d_Rarray[i]);
		cudaErr = cudaFree(h_d_Distarray[i]);
	}

	free(Aarray);
	Aarray = NULL;

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

	cudaErr = cudaFree(dev_hypox);
	dev_hypox = NULL;
	cudaErr = cudaFree(dev_hypoy);
	dev_hypoy = NULL;

	cudaErr = cudaFree(d_Amat);
	d_Amat = NULL;

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

	//2. Resign new spaces:
	h_d_Aarray = (double**)malloc(num * sizeof(double*));
	h_d_Tauarray = (double**)malloc(num * sizeof(double*));
	h_d_Qarray = (double**)malloc(num * sizeof(double*));
	h_d_Rarray = (double**)malloc(num * sizeof(double*));
	h_d_Paraarray = (double**)malloc(num * sizeof(double*));

	for (int i = 0; i < num; i++)
	{
		rows = InlierNum[i];
		cudaErr = cudaMalloc((void**)&h_d_Aarray[i], rows*cols * sizeof(double));
		cudaErr = cudaMalloc((void**)&h_d_Tauarray[i], ltau * sizeof(double));
		cudaErr = cudaMalloc((void**)&h_d_Qarray[i], rows*rows * sizeof(double));
		cudaErr = cudaMalloc((void**)&h_d_Rarray[i], rows*cols * sizeof(double));
		cudaErr = cudaMalloc((void**)&h_d_Paraarray[i], cols * sizeof(double));
	}

	//Copy the host array of device pointers to the device
	stat = cublasAlloc(num, sizeof(double*), (void**)&d_Aarray);
	stat = cublasAlloc(num, sizeof(double*), (void**)&d_Tauarray);
	cudaErr = cudaMalloc((void**)&d_Qarray, num * sizeof(double*));
	cudaErr = cudaMalloc((void**)&d_Rarray, num * sizeof(double*));
	cudaErr = cudaMalloc((void**)&d_Paraarray, num * sizeof(double*));

	cudaErr = cudaMemcpy(d_Qarray, h_d_Qarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Rarray, h_d_Rarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Paraarray, h_d_Paraarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	
	SetupMatricesOnDevice<<<(maxIters+255)/256,256>>>(xvals, maxIters, pcsize, parasize, d_inlierNum, d_Inliers, d_Aarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();

	// GetModelParawithVariateAs(double** Amat, size_t pcsize, int* inlierNum, int maxIt, int paraSize,
	//	double** inlier_ys, double** paras)
	GetModelParawithVariateAs<<<(maxIters+255)/256,256>>>(d_Aarray, pcsize, d_inlierNum, maxIters, parasize, d_inliery, d_Paraarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();

	GetModelSqDist << <(maxIters + 255) / 256, 256 >> > (dev_xs, dev_ys, pcsize, maxIters, parasize, d_Paraarray, d_ModelErr, d_Distarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaMemcpy(h_d_Distarray, d_Distarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(modelErr, d_ModelErr, num * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaMemcpy(dists[i], h_d_Distarray[i], pcsize * sizeof(double), cudaMemcpyDeviceToHost);
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
	}

	//Host:
	free(hst_hypoIDs);
	hst_hypoIDs = NULL;

	free(Aarray);
	Aarray = NULL;

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
	
	//Device space:
	cudaErr = cudaFree(dev_hypox);
	dev_hypox = NULL;
	cudaErr = cudaFree(dev_hypoy);
	dev_hypoy = NULL;

	cudaErr = cudaFree(d_Amat);
	d_Amat = NULL;

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