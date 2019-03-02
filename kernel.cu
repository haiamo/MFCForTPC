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

extern "C" cudaError_t RANSACOnGPU(double* xvals, double* yvals, size_t pcsize, int maxIters, int minInliers, int parasize,
	double* &hst_hypox, double* &hst_hypoy, double* &As, double** &Qs, double** &taus, double** &Rs, double**& paras,
	double* &modelErr, double** &dists)
{
	//Prepare device space and parameters.
	cudaError_t cudaErr;
	cublasStatus_t stat;
	double /* *dev_x, *dev_y,*/*dev_hypox=NULL, *dev_hypoy;
	unsigned int /* *dev_hypoIDs,*/ *hst_hypoIDs;

	//cudaMalloc((void**)&dev_x, sizeof(double)*pcsize);//PointCloud Size
	//cudaMalloc((void**)&dev_y, sizeof(double)*pcsize);//PointCloud Size
	//cudaMemcpy(dev_x, xvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_y, yvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);
	//cudaMalloc((void**)&dev_para, sizeof(double)*maxIters*parasize);//MaxIterations*(Order+1)
	//cudaMalloc((void**)&dev_dist, sizeof(double)*maxIters*pcsize);//MaxIterations*PointCloud Size

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

	hst_hypoIDs = (unsigned int *)malloc(sizeof(unsigned int)*maxIters*minInliers);

	//Generate random hypo-inliers IDs.
	srand(unsigned(time(NULL)));
	for (size_t ii = 0; ii < maxIters*minInliers; ii++)
	{
		hst_hypoIDs[ii] = rand() % pcsize;
		hst_hypox[ii] = xvals[hst_hypoIDs[ii]];
		hst_hypoy[ii] = yvals[hst_hypoIDs[ii]];
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
	//Retrieve result matrix from device
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
	//Free spaces
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

	return cudaErr;
}