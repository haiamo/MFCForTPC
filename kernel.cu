#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "cuda.h"
#include "device_functions.h"
#include "cublas.h"
#include "cublas_v2.h"
#include "cudaMain.h"
#include "stdio.h"
 
inline void checkCudaErrors(cudaError err)//错误处理函数
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA Runtime API error: %s.\n", cudaGetErrorString(err));
		return;
	}
}
 
__global__ void add(double *a, double *b, double *c)//处理核函数
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	//for (size_t k = 0; k < 50000; k++)
	{
		c[tid] = a[tid] * sin(b[tid]);
	}
}
 
extern "C" int runtest(double *host_a, double *host_b, double *host_c, int cursize)
{
	double *dev_a, *dev_b, *dev_c;
 
	checkCudaErrors(cudaMalloc((void**)&dev_a, sizeof(double)* cursize));//分配显卡内存
	checkCudaErrors(cudaMalloc((void**)&dev_b, sizeof(double)* cursize));
	checkCudaErrors(cudaMalloc((void**)&dev_c, sizeof(double)* cursize));
 
	checkCudaErrors(cudaMemcpy(dev_a, host_a, sizeof(double)* cursize, cudaMemcpyHostToDevice));//将主机待处理数据内存块复制到显卡内存中
	checkCudaErrors(cudaMemcpy(dev_b, host_b, sizeof(double)* cursize, cudaMemcpyHostToDevice));
 
	add <<<(cursize+255) / 256, 256 >>>(dev_a, dev_b, dev_c);//调用显卡处理数据
	checkCudaErrors(cudaMemcpy(host_c, dev_c, sizeof(double)* cursize, cudaMemcpyDeviceToHost));//将显卡处理完数据拷回来
 
	cudaFree(dev_a);//清理显卡内存
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}

__global__ void GPUCharKernel(char** in_line, float* dev_x, float* dev_y, float* dev_z,size_t in_height, size_t in_len)
{
	int value_pos = 0, in_id = 0;
	float cur_val = 0.0f, factor = 10.0f, sign = 1.0f;
	bool bValStart = false;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid < in_height)
	{
		while(in_id <= in_len)
		{
			if (in_line[tid][in_id] == ' ' || in_line[tid][in_id] == '\n' || in_line[tid][in_id] == '\0' || in_id == in_len)
			{
				if (bValStart)
				{
					cur_val *= sign;
					switch (value_pos)
					{
					case 0:
						dev_x[tid] = cur_val;
						break;
					case 1:
						dev_y[tid] = cur_val;
						break;
					case 2:
						dev_z[tid] = cur_val;
						break;
					}
					value_pos++;
					sign = 1.0f;
					factor = 10.0f;
					cur_val = 0.0;
					bValStart = false;
				}
			}
			else
			{
				if (in_line[tid][in_id] == '-')
				{
					sign = -1.0f;
				}
				else if(in_line[tid][in_id]=='.')
				{
					factor = 1.0f;
				}
				else
				{
					if (factor > 1.0f)
					{
						cur_val = cur_val * 10 + ((int)in_line[tid][in_id]-(int)'0');
					}
					else
					{
						factor *= 0.1f;
						cur_val += factor*((int)in_line[tid][in_id]-(int)'0');
					}
				}

				bValStart = true;
			}
			in_id++;
		};
	}

}

__global__ void PCDistanceKernel(double* xval, double* yval, double* paras, int offset, double* dist)
{
	int tid = offset + blockIdx.x*blockDim.x + threadIdx.x;
	double xx , yy ,curD=0.0, a = paras[3], b = paras[2], c = paras[1], d = paras[0];
	xx = xval[tid]*1.0;
	yy = yval[tid]*1.0;
	Eigen::Vector3d iniPt, k_thPt, k_thF, k_thDeltX;
	iniPt(0) = xx;
	iniPt(1) = yy;
	iniPt(2) = 1.0;
	k_thPt = iniPt;
	Eigen::Matrix3d k_thJ, k_thJAdj;
	double eps = 10e-5, g1 = 0.0, g2 = 0.0, detJ = 0.0;
	int order = 3;
	do
	{
		if (order == 3)
		{
			k_thF(0) = -1.0*(3 * a*k_thPt(2)*k_thPt(0)*k_thPt(0) + 2 * b*k_thPt(2)*k_thPt(0) + 2 * k_thPt(0) + c*k_thPt(2) - 2 * iniPt(0));
			k_thF(1) = -1.0*(2 * k_thPt(1) - 2 * iniPt(1) - k_thPt(2));
			k_thF(2) = -1.0*(a*k_thPt(0)*k_thPt(0)*k_thPt(0) + b*k_thPt(0)*k_thPt(0) + c*k_thPt(0) + d - k_thPt(1));

			g1 = 6 * a*k_thPt(0)*k_thPt(2) + 2 * b*k_thPt(2) + 2;
			g2 = 3 * a*k_thPt(0)*k_thPt(0) + 2 * b*k_thPt(0) + c;

			k_thJ(0, 0) = g1; k_thJ(0, 1) = 0.0; k_thJ(0, 2) = g2;
			k_thJ(1, 0) = 0.0; k_thJ(1, 1) = 2.0; k_thJ(1, 2) = -1.0;
			k_thJ(2, 0) = g2; k_thJ(2, 1) = -1.0; k_thJ(2, 2) = 0.0;

			k_thJAdj(0, 0) = -1; k_thJAdj(0, 1) = -g2; k_thJAdj(0, 2) = -2 * g2;
			k_thJAdj(1, 0) = -g2; k_thJAdj(1, 1) = -g2 * g2; k_thJAdj(1, 2) = g1;
			k_thJAdj(2, 0) = -2 * g2; k_thJAdj(2, 1) = g1; k_thJAdj(2, 2) = 2 * g1;

			detJ = -2 * g2 * g2 - g1;
			if (abs(detJ) < eps)
			{
				detJ = 1e-5;
			}
		}
		//k_thDeltX = (k_thJ.householderQr()).solve(k_thF);
		k_thDeltX = -1.0 / detJ*k_thJAdj*k_thF;
		
		k_thPt += k_thDeltX;
	} while ((abs(k_thF(0)) >= eps && abs(k_thF(1)) >= eps && abs(k_thF(2)) >= eps) &&
		(abs(k_thDeltX(0)) >= eps && abs(k_thDeltX(1)) >= eps && abs(k_thDeltX(2)) >= eps));
	curD = sqrt((k_thPt(0) - iniPt(0))*(k_thPt(0) - iniPt(0)) + (k_thPt(0) - iniPt(0))*(k_thPt(0) - iniPt(0)));
	dist[tid] = curD;
}

__global__ void PCInterceptKernel(double* xval, double* yval, double* paras, int offset, double* dist)
{
	int tid = offset + blockIdx.x*blockDim.x + threadIdx.x;
	double x = xval[tid], y = yval[tid] , curD = 0.0, a = paras[3], b = paras[2], c = paras[1], d = paras[0];
	curD = y - a*x*x*x - b*x*x - c*x - d;
	dist[tid] = curD;
}

extern "C" void GPUCharToValue(char** in_chars, float* host_x, float* host_y, float* host_z, size_t height, size_t width)
{
	char** h_char2d = (char**)malloc(height * sizeof(char*));
	char** d_char2d;
	char* d_char1d;
	checkCudaErrors(cudaMalloc((void**)&d_char1d, height*width * sizeof(char)));
	for (int ii = 0; ii < height; ii++)
	{
		h_char2d[ii] = d_char1d + ii*width;
		checkCudaErrors(cudaMemcpy(h_char2d[ii], in_chars[ii], width * sizeof(char), cudaMemcpyHostToDevice));
	}
	checkCudaErrors(cudaMalloc((void**)&d_char2d, sizeof(char*)*height));
	checkCudaErrors(cudaMemcpy(d_char2d, h_char2d, sizeof(char*)*height, cudaMemcpyHostToDevice));

	float* dev_x, *dev_y, *dev_z;
	checkCudaErrors(cudaMalloc((void**)&dev_x, sizeof(float)* height));
	checkCudaErrors(cudaMalloc((void**)&dev_y, sizeof(float)* height));
	checkCudaErrors(cudaMalloc((void**)&dev_z, sizeof(float)* height));
	
	GPUCharKernel <<<(height+255) / 256, 256 >>> (d_char2d, dev_x,dev_y,dev_z,height, width);
	cudaDeviceSynchronize();

	cudaMemcpy(host_x, dev_x, sizeof(float)* height, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_y, dev_y, sizeof(float)* height, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_z, dev_z, sizeof(float)* height, cudaMemcpyDeviceToHost);

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_z);
	cudaFree(d_char2d);
	for (int ii = 0; ii < height; ii++)
	{
		cudaFree(h_char2d[ii]);
	}
	
	cudaFree(d_char1d);
	free(h_char2d);
}

extern "C" void GetPCDistance(double * xvals, double * yvals, size_t pcsize, double * paras, int parasize, double * dists)
{
	double *dev_x, *dev_y, *dev_para, *dev_dist;
	cudaMalloc((void**)&dev_x, sizeof(double)*pcsize);
	cudaMalloc((void**)&dev_y, sizeof(double)*pcsize);
	cudaMalloc((void**)&dev_para, sizeof(double)*parasize);
	cudaMalloc((void**)&dev_dist, sizeof(double)*pcsize);

	cudaMemcpy(dev_x, xvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, yvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_para, paras, sizeof(double)*parasize, cudaMemcpyHostToDevice);

	PCDistanceKernel <<<(pcsize + 255) / 256, 256 >>> (dev_x, dev_y, dev_para, 0, dev_dist);

	cudaMemcpy(dists, dev_dist, sizeof(double)*pcsize, cudaMemcpyDeviceToHost);

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_dist);
	cudaFree(dev_para);
}

extern "C" void GetPCDistanceAsynch(double* xvals, double* yvals, size_t pcsize, double* paras, int parasize, double* dists)
{
	//Asynchoronous Method
	double *dev_x, *dev_y, *dev_para, *dev_dist;
	const int nStream = 5;
	int streamSize = pcsize / nStream + 1;
	int lastStreamSize = fmod(pcsize, nStream);
	if (abs(fmod(pcsize, nStream)) < 1e-5)
	{
		streamSize = pcsize / nStream;
		lastStreamSize = streamSize;
	}
	else
	{
		streamSize = pcsize / nStream + 1;
		lastStreamSize =(int) fmod(pcsize, nStream);
	}
	cudaStream_t streams[nStream];
	for (int ii = 0; ii < nStream; ++ii)
	{
		cudaStreamCreate(&streams[ii]);
	}

	cudaMalloc((void**)&dev_x, sizeof(double)*pcsize);
	cudaMalloc((void**)&dev_y, sizeof(double)*pcsize);
	cudaMalloc((void**)&dev_para, sizeof(double)*parasize);
	cudaMalloc((void**)&dev_dist, sizeof(double)*pcsize);
	int curStreamSize = streamSize;
	for (int ii = 0; ii < nStream; ++ii)
	{
		int offset = ii*streamSize;
		if (ii == nStream - 1)
			curStreamSize = lastStreamSize;
		cudaMemcpyAsync(&dev_x[offset], &xvals[offset], sizeof(double)*curStreamSize, cudaMemcpyHostToDevice, streams[ii]);
		cudaMemcpyAsync(&dev_y[offset], &yvals[offset], sizeof(double)*curStreamSize, cudaMemcpyHostToDevice, streams[ii]);
		cudaMemcpyAsync(dev_para, paras, sizeof(double)*parasize, cudaMemcpyHostToDevice, streams[ii]);

		PCDistanceKernel <<<(curStreamSize+255) / 256, 256, 0, streams[ii] >>> (dev_x, dev_y, dev_para, offset, dev_dist);

		cudaMemcpyAsync(&dists[offset], &dev_dist[offset], sizeof(double)*curStreamSize, cudaMemcpyDeviceToHost, streams[ii]);
	}

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_para);
	cudaFree(dev_dist);
	for (int ii = 0; ii < nStream; ii++)
	{
		cudaStreamDestroy(streams[ii]);
	}
}

extern "C" void GetPCIntercept(double * xvals, double * yvals, size_t pcsize, double * paras, int parasize, double * dists)
{
	double *dev_x, *dev_y, *dev_para, *dev_dist;
	cudaMalloc((void**)&dev_x, sizeof(double)*pcsize);
	cudaMalloc((void**)&dev_y, sizeof(double)*pcsize);
	cudaMalloc((void**)&dev_para, sizeof(double)*parasize);
	cudaMalloc((void**)&dev_dist, sizeof(double)*pcsize);

	cudaMemcpy(dev_x, xvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, yvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_para, paras, sizeof(double)*parasize, cudaMemcpyHostToDevice);

	PCInterceptKernel <<<(pcsize + 255) / 256, 256 >>> (dev_x, dev_y, dev_para, 0, dev_dist);

	cudaMemcpy(dists, dev_dist, sizeof(double)*pcsize, cudaMemcpyDeviceToHost);

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_dist);
	cudaFree(dev_para);
}

extern "C" void GetPCInterceptAsynch(double* xvals, double* yvals, size_t pcsize, double* paras, int parasize, double* dists)
{
	//Asynchoronous Method
	double *dev_x, *dev_y, *dev_para, *dev_dist;
	const int nStream = 5;
	int streamSize = pcsize / nStream + 1;
	int lastStreamSize = fmod(pcsize, nStream);
	if (abs(fmod(pcsize, nStream)) < 1e-5)
	{
		streamSize = pcsize / nStream;
		lastStreamSize = streamSize;
	}
	else
	{
		streamSize = pcsize / nStream + 1;
		lastStreamSize = (int)fmod(pcsize, nStream);
	}
	cudaStream_t streams[nStream];
	for (int ii = 0; ii < nStream; ++ii)
	{
		cudaStreamCreate(&streams[ii]);
	}

	cudaMalloc((void**)&dev_x, sizeof(double)*pcsize);
	cudaMalloc((void**)&dev_y, sizeof(double)*pcsize);
	cudaMalloc((void**)&dev_para, sizeof(double)*parasize);
	cudaMalloc((void**)&dev_dist, sizeof(double)*pcsize);
	int curStreamSize = streamSize;
	for (int ii = 0; ii < nStream; ++ii)
	{
		int offset = ii*streamSize;
		if (ii == nStream - 1)
			curStreamSize = lastStreamSize;
		cudaMemcpyAsync(&dev_x[offset], &xvals[offset], sizeof(double)*curStreamSize, cudaMemcpyHostToDevice, streams[ii]);
		cudaMemcpyAsync(&dev_y[offset], &yvals[offset], sizeof(double)*curStreamSize, cudaMemcpyHostToDevice, streams[ii]);
		cudaMemcpyAsync(dev_para, paras, sizeof(double)*parasize, cudaMemcpyHostToDevice, streams[ii]);

		PCInterceptKernel <<<(curStreamSize + 255) / 256, 256, 0, streams[ii] >>> (dev_x, dev_y, dev_para, offset, dev_dist);

		cudaMemcpyAsync(&dists[offset], &dev_dist[offset], sizeof(double)*curStreamSize, cudaMemcpyDeviceToHost, streams[ii]);
	}

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_para);
	cudaFree(dev_dist);
	for (int ii = 0; ii < nStream; ii++)
	{
		cudaStreamDestroy(streams[ii]);
	}
}

__global__ void SetupRand_Kernel(curandState *state, time_t curT)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	curand_init(curT, tid, 0, &state[tid]);
}

__global__ void GenerateRand_Kernel(curandState *state, size_t pcsize, double* in_x, double* in_y, unsigned int *hypoIds, double *hypox, double *hypoy)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	//int tidx = gridDim.x*blockIdx.x + threadIdx.x;
	//int tidy = gridDim.y*blockIdx.y + threadIdx.y;
	unsigned int x = curand(&state[tid]) % pcsize;
	hypoIds[tid] = x;
	hypox[tid] = in_x[x];
	hypoy[tid] = in_y[x];
	int set = 0;
}

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

__global__ void SetupMatricesPitch(double* xvals, double* yvals, int maxIt, int inliers, int paras, size_t pitch, double* Amat)
{
	int BlockID = blockIdx.x + blockIdx.y*gridDim.x;
	int tid = BlockID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	double* p_line = (double*)((char*)Amat + tid*pitch);
	if (tid < maxIt)
	{
		for (int Acoli = 0; Acoli < paras; Acoli++)
		{
			for (int ii = 0; ii < inliers; ii++)
			{
				if (Acoli == 0)
				{
					p_line[ii] = 1;
				}
				else
				{
					p_line[Acoli*inliers + ii] = p_line[(Acoli - 1)*inliers + ii] * xvals[tid*inliers + ii];
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
	//cudaMalloc((void**)&dev_hypoIDs, sizeof(unsigned int)*maxIters*minInliers);//MaxIterations*MinInliers matrix of point ID.

	//Generate random hypo-inliers IDs.
	/*dim3 grids((maxIters*minInliers + 255) / 256);
	dim3 blocks(256);
	curandState *cuStates;
	cudaMalloc((void**)&cuStates, sizeof(curandState)*grids.x*grids.y*blocks.x*blocks.y);

	SetupRand_Kernel <<<grids,blocks >>> (cuStates, time(NULL));

	GenerateRand_Kernel <<<grids, blocks >>> (cuStates, pcsize, dev_x, dev_y, dev_hypoIDs, dev_hypox, dev_hypoy);

	cudaMemcpy(hst_hypoIDs, dev_hypoIDs, sizeof(unsigned int)*maxIters*minInliers, cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_hypox, dev_hypox, sizeof(double)*maxIters*minInliers, cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_hypoy, dev_hypoy, sizeof(double)*maxIters*minInliers, cudaMemcpyDeviceToHost);*/
	
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
	//h_Amat = (double*)malloc(sizeof(double*)*num*rows*cols);
	//memset(h_Amat, -1.0, sizeof(double)*num*rows*cols);
	//Setup device matrices

	//SetupMatricesPitch << <(maxIters + 255) / 256, 256 >> > (dev_hypox, dev_hypoy, maxIters, minInliers, parasize, pitchA, Amat);
	//cublasGetVector(num*rows*cols, sizeof(double), Amat, 1, h_Amat, 1);
	//cudaMemcpy2D(h_Amat, sizeof(double)*rows*cols, Amat, pitchA, sizeof(double)*rows*cols, num, cudaMemcpyDeviceToHost);
	//cudaMemcpy(hst_hypox, dev_hypox, sizeof(double)*num*rows, cudaMemcpyDeviceToHost);
	//cudaMemcpy(hst_hypoy, dev_hypoy, sizeof(double)*num*rows, cudaMemcpyDeviceToHost);
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
	//Tauarray = (double**)malloc(num * sizeof(double*));
	//Qarray = (double**)malloc(num * sizeof(double*));
	//Rarray = (double**)malloc(num * sizeof(double*));

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
		//Tauarray[i] = (double*)malloc(ltau * sizeof(double));
		//Qarray[i] = (double*)malloc(rows*rows * sizeof(double));
		//[i] = (double*)malloc(rows*cols * sizeof(double));
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
	cudaErr = cudaMalloc((void**)&d_ModelErr, pcsize * sizeof(double));
	
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaMalloc((void**)&h_d_Aarray[i], rows*cols * sizeof(double));
		cudaErr = cudaMalloc((void**)&h_d_Tauarray[i], ltau * sizeof(double));
		//stat = cublasAlloc(rows*cols, sizeof(double), (void**)&h_d_Aarray[i]);
		//stat = cublasAlloc(ltau, sizeof(double), (void**)&h_d_Tauarray[i]);
		cudaErr = cudaMalloc((void**)&h_d_Qarray[i], rows*rows * sizeof(double));
		cudaErr = cudaMemcpy(h_d_Qarray[i], Qarray[i], sizeof(double)*rows*rows, cudaMemcpyHostToDevice);
		cudaErr = cudaMalloc((void**)&h_d_Rarray[i], rows*cols * sizeof(double));
		cudaErr = cudaMalloc((void**)&h_d_Paraarray[i], cols * sizeof(double));
		cudaErr = cudaMalloc((void**)&h_d_Distarray[i], pcsize * sizeof(double));
	}

	//Copy the host array of device pointers to the device
	//cudaErr = cudaMalloc((void**)&d_Aarray, num * sizeof(double*));
	//cudaErr = cudaMalloc((void**)&d_Tauarray, num * sizeof(double*));
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
		//stat = cublasSetMatrix(rows, cols, sizeof(double), Aarray[i], rows, h_d_Aarray[i], rows);
		//stat = cublasSetVector(ltau, sizeof(double), Tauarray[i], 1, h_d_Tauarray[i], 1);
		cudaErr = cudaMemcpy(h_d_Aarray[i], Aarray[i], rows*cols * sizeof(double), cudaMemcpyHostToDevice);
		cudaErr = cudaMemcpy(h_d_Tauarray[i], Tauarray[i], ltau * sizeof(double), cudaMemcpyHostToDevice);
		//cudaErr = cudaMemcpy(d_Aarray[i], Aarray[i], rows*cols * sizeof(double), cudaMemcpyHostToDevice);
		//cudaErr = cudaMemcpy(d_Tauarray[i], Tauarray[i], ltau * sizeof(double), cudaMemcpyHostToDevice);
	}
	//stat = cublasSetVector(num, sizeof(double*), h_d_Tauarray,1, d_Tauarray,1 );
	//stat = cublasSetVector(num, sizeof(double*), h_d_Aarray, 1, d_Aarray, 1);
	cudaErr = cudaMemcpy(d_Aarray, h_d_Aarray, num *sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(d_Tauarray, h_d_Tauarray, num *sizeof(double*), cudaMemcpyHostToDevice);
	cudaErr = cudaDeviceSynchronize();
	int *info, lda = rows;
	//cudaErr = cudaMalloc((void**)&info, sizeof(int)*num);
	info = (int*)malloc(sizeof(int)*num);
	stat = cublasDgeqrfBatched(handle, rows, cols, d_Aarray, rows, d_Tauarray, info, num);
	cudaErr = cudaDeviceSynchronize();
	//Retrieve result matrix from device
	cudaErr = cudaMemcpy(h_d_Aarray, d_Aarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(h_d_Tauarray, d_Tauarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	for (int i = 0; i < num; i++)
	{
		//stat = cublasGetMatrix(rows, cols, sizeof(double), h_d_Aarray[i], rows, Aarray[i], rows);
		
		//stat = cublasGetVector(ltau, sizeof(double), h_d_Tauarray[i], 1, Tauarray[i], 1);
		cudaErr = cudaMemcpy(Aarray[i], h_d_Aarray[i], rows*cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaErr = cudaMemcpy(Tauarray[i], h_d_Tauarray[i], ltau * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaErr = cudaMemcpy(Aarray[i], d_Aarray[i], rows*cols * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaErr = cudaMemcpy(Tauarray[i], d_Tauarray[i], ltau * sizeof(double), cudaMemcpyDeviceToHost);
		//cudaErr = cudaMemcpy(h_dd_Aarray[i], Aarray[i], rows*cols * sizeof(double), cudaMemcpyHostToDevice);
		//cudaErr = cudaMemcpy(h_dd_Tauarray[i], Tauarray[i], ltau * sizeof(double), cudaMemcpyHostToDevice);

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

	//cublasGetVector(num, sizeof(double*), d_Tauarray, 1, Tauarray, 1);
	//cudaErr = cudaMemcpy(dd_Aarray, h_dd_Aarray, num * sizeof(double*), cudaMemcpyHostToDevice);
	//cudaErr = cudaMemcpy(dd_Tauarray, h_dd_Tauarray, num * sizeof(double*), cudaMemcpyHostToDevice);

	//GetModelPara(cublasHandle_t handle, double** QRs, double** taus, double* hypoy, int maxIt,
	//	int inliers/*rows, int parasize/*columns, double** Q, double** R, double** paras)
	
	GetModelPara <<<(maxIters + 255) / 256, 256>>>(d_Aarray, d_Tauarray, dev_hypoy, num,
		rows, cols, d_Qarray, d_Rarray, d_Paraarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaMemcpy(h_d_Qarray, d_Qarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(h_d_Rarray, d_Rarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(h_d_Paraarray, d_Paraarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaDeviceSynchronize();
	//cudaErr = cudaDeviceSynchronize();
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaMemcpy(Qarray[i], h_d_Qarray[i], rows*rows * sizeof(double), cudaMemcpyDeviceToHost);
		cudaErr = cudaMemcpy(Rarray[i], h_d_Rarray[i], rows*cols * sizeof(double), cudaMemcpyDeviceToHost);
		cudaErr = cudaMemcpy(Paraarray[i], h_d_Paraarray[i], cols * sizeof(double), cudaMemcpyDeviceToHost);
	}
	//cudaErr = cudaThreadSynchronize();
	cudaErr = cudaDeviceSynchronize();

	//Get model error and point to curve distances.
	//__global__ void GetModelSqDist(double* xs, double* ys,size_t pcsize, int maxIt, int parasize, double** paras,
	//								double* modelErr, double** dists)
	double* dev_xs, *dev_ys;
	cudaErr = cudaMalloc((void**)&dev_xs, sizeof(double)*pcsize);
	cudaErr = cudaMalloc((void**)&dev_ys, sizeof(double)*pcsize);
	cudaErr = cudaMemcpy(dev_xs, xvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);
	cudaErr = cudaMemcpy(dev_ys, yvals, sizeof(double)*pcsize, cudaMemcpyHostToDevice);

	GetModelSqDist <<<(maxIters + 255) / 256, 256 >>> (dev_xs, dev_ys, pcsize, maxIters, parasize, d_Paraarray, d_ModelErr, d_Distarray);
	cudaErr = cudaGetLastError();
	cudaErr = cudaDeviceSynchronize();
	cudaErr = cudaMemcpy(h_d_Distarray, d_Distarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
	cudaErr = cudaMemcpy(modelErr, d_ModelErr, pcsize * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < num; i++)
	{
		cudaErr = cudaMemcpy(dists[i], h_d_Distarray[i], pcsize * sizeof(double), cudaMemcpyDeviceToHost);
	}
	//Free spaces
	//Host-device pointer:
	for (int i = 0; i < num; i++)
	{
		//free(Aarray[i]);
		//free(Tauarray[i]);
		cudaErr = cudaFree(h_d_Aarray[i]);
		cudaErr = cudaFree(h_d_Tauarray[i]);
		cudaErr = cudaFree(h_d_Qarray[i]);
		cudaErr = cudaFree(h_d_Rarray[i]);
		cudaErr = cudaFree(h_d_Distarray[i]);
	}

	//Host:
	free(hst_hypoIDs);
	hst_hypoIDs = NULL;
	//free(h_Amat);

	//free(info);
	//info = NULL;

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


	//cublasDestroy(handle);
	return cudaErr;
}