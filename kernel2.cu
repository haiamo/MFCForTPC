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

#define cudaERRORHANDEL(err, info) {if(cudaSuccess!=(err))\
									{return GetCUDAErrorInfo((err),(info));}}

__host__ __device__ TPCVec4::TPCVec4(const Point3D& inVec, const float wt)
{
	x = inVec.x3;
	y = inVec.y3;
	z = inVec.z3;
	w = wt;
}

__host__ __device__ TPCVec4::~TPCVec4()
{
}

__device__ TPCVec4 TPCVec4::HomoCoordPtToPt3Dw()
{
	TPCVec4 tmp;
	if (w > 1e-6 || w < -1e-6)
	{
		tmp.x = x / w;
		tmp.y = y / w;
		tmp.z = z / w;
		tmp.w = w;
	}
	else
	{
		tmp.x = x;
		tmp.y = y;
		tmp.z = z;
		tmp.w = w;
	}
	return tmp;
}

__device__ TPCVec4 TPCVec4::Pt3DwToHomoCoordPt()
{
	TPCVec4 tmp;
	if (w > 1e-6 || w < -1e-6)
	{
		tmp.x = x * w;
		tmp.y = y * w;
		tmp.z = z * w;
		tmp.w = w;
	}
	else
	{
		tmp.x = x;
		tmp.y = y;
		tmp.z = z;
		tmp.w = 0.0;
	}
	return tmp;
}


__device__ Point3D TPCVec4::HomoCoordPtToPt3D()
{
	Point3D tmp;
	if (w > 1e-6 || w < -1e-6)
	{
		tmp.x3 = x / w;
		tmp.y3 = y / w;
		tmp.z3 = z / w;
	}
	else
	{
		tmp.x3 = x;
		tmp.y3 = y;
		tmp.z3 = z;
	}
	return tmp;
}

__device__ TPCVec4 TPCVec4::Pt3DToHomoCoordPt(Point3D inV, float wt)
{
	TPCVec4 tmp;
	if (wt > 1e-6 || wt < -1e-6)
	{
		tmp.x = x * wt;
		tmp.y = y * wt;
		tmp.z = z * wt;
		tmp.w = wt;
	}
	else
	{
		tmp.x = x;
		tmp.y = y;
		tmp.z = z;
		tmp.w = 0.0;
	}
	return tmp;
}

__device__ void TPCVec4::GetPoint3D(Point3D& outP3D)
{
	outP3D.x3 = x;
	outP3D.y3 = y;
	outP3D.z3 = z;
}

__device__ TPCVec4 TPCVec4::Cross(TPCVec4 v)
{
	TPCVec4 tmp;
	tmp.x = x*v.y - y*v.x;
	tmp.y = -(x*v.z - z*v.x);
	tmp.z = x*v.y - y*v.x;
	return tmp;
}

__device__ double TPCVec4::Dot(TPCVec4 v)
{
	double tmp;
	tmp = x*v.x + y*v.y + z*v.z;
	return tmp;
}

__device__ double TPCVec4::Length()
{
	double tmp;
	tmp = sqrt(x*x + y*y + z*z);
	return tmp;
}


__host__ __device__ TPCVec4 TPCVec4::operator=(const TPCVec4 & inVec4)
{
	if (this != &inVec4)
	{
		x = inVec4.x;
		y = inVec4.y;
		z = inVec4.z;
		w = inVec4.w;
	}
	return *this;
}

__device__ TPCVec4 operator+(const TPCVec4 &vec1, const TPCVec4 &vec2)
{
	TPCVec4 tmp;
	tmp.x = vec1.x + vec2.x;
	tmp.y = vec1.y + vec2.y;
	tmp.z = vec1.z + vec2.z;
	tmp.w = vec1.w + vec2.w;
	return tmp;
}

__device__ TPCVec4 operator+=(TPCVec4 &vec1, const TPCVec4 &vec2)
{
	vec1.x += vec2.x;
	vec1.y += vec2.y;
	vec1.z += vec2.z;
	vec1.w += vec2.w;
	return vec1;
}

__device__ TPCVec4 operator-(const TPCVec4 &vec1, const TPCVec4 &vec2)
{
	TPCVec4 tmp;
	tmp.x = vec1.x - vec2.x;
	tmp.y = vec1.y - vec2.y;
	tmp.z = vec1.z - vec2.z;
	tmp.w = vec1.w - vec2.w;
	return tmp;
}

__device__ TPCVec4 operator-=(TPCVec4 &vec1, const TPCVec4 &vec2)
{
	vec1.x -= vec2.x;
	vec1.y -= vec2.y;
	vec1.z -= vec2.z;
	vec1.w -= vec2.w;
	return vec1;
}

__device__ TPCVec4 operator*(const TPCVec4 &vec, const float &val)
{
	TPCVec4 tmp;
	tmp.x = vec.x * val;
	tmp.y = vec.y * val;
	tmp.z = vec.z * val;
	tmp.w = vec.w * val;
	return tmp;
}

__device__ TPCVec4 operator*(const float &val, const TPCVec4 &vec)
{
	TPCVec4 tmp;
	tmp.x = vec.x * val;
	tmp.y = vec.y * val;
	tmp.z = vec.z * val;
	tmp.w = vec.w * val;
	return tmp;
}

__device__ TPCVec4 operator*=(TPCVec4 &vec, const float &val)
{
	vec.x *= val;
	vec.y *= val;
	vec.z *= val;
	vec.w *= val;
	return vec;
}

__device__ TPCVec4 operator/(const TPCVec4 &vec, const float &val)
{
	TPCVec4 tmp;
	tmp.x = vec.x / val;
	tmp.y = vec.y / val;
	tmp.z = vec.z / val;
	tmp.w = vec.w / val;
	return tmp;
}

__device__ TPCVec4 operator/=(TPCVec4 &vec, const float &val)
{
	vec.x /= val;
	vec.y /= val;
	vec.z /= val;
	vec.w /= val;
	return vec;
}

__device__ Point3D P3DCross(Point3D pt1, Point3D pt2)
{
	Point3D tmp;
	tmp.x3 = pt1.y3*pt2.z3 - pt1.z3*pt2.y3;
	tmp.y3 = -(pt1.x3*pt2.z3 - pt1.z3*pt2.x3);
	tmp.z3 = pt1.x3*pt2.y3 - pt1.y3*pt2.x3;
	return tmp;
}

__device__ double P3DDot(Point3D pt1, Point3D pt2)
{
	double tmp;
	tmp = pt1.x3*pt2.x3 + pt1.y3*pt2.y3 + pt1.z3*pt2.z3;
	return tmp;
}

__device__ double P3DLen(Point3D pt)
{
	double tmp;
	tmp = sqrt(pt.x3*pt.x3 + pt.y3*pt.y3 + pt.z3*pt.z3);
	return tmp;
}

__device__ Point3D P3DSub(Point3D pt1, Point3D pt2)
{
	Point3D tmp;
	tmp.x3 = pt1.x3 - pt2.x3;
	tmp.y3 = pt1.y3 - pt2.y3;
	tmp.z3 = pt1.z3 - pt2.z3;
	return tmp;
}

__device__ Point3D P3DSum(Point3D pt1, Point3D pt2)
{
	Point3D tmp;
	tmp.x3 = pt1.x3 + pt2.x3;
	tmp.y3 = pt1.y3 + pt2.y3;
	tmp.z3 = pt1.z3 + pt2.z3;
	return tmp;
}

__device__ Point3D P3DMult(Point3D pt, float val)
{
	Point3D tmp;
	tmp.x3 = pt.x3*val;
	tmp.y3 = pt.y3*val;
	tmp.z3 = pt.z3*val;
	return tmp;
}

__device__ Point3D P3DDev(Point3D pt, float val)
{
	Point3D tmp = pt;
	if (abs(val) > 1e-5)
	{
		tmp.x3 = pt.x3 / val;
		tmp.y3 = pt.y3 / val;
		tmp.z3 = pt.z3 / val;
	}
	return tmp;
}

__host__ __device__ bool dIsZero(double inVal)
{
	if (inVal*1.0<1e-6 && inVal*1.0>-1e-6)
	{
		return true;
	}
	else
	{
		return false;
	}
}

__global__ void GetOneLineData(__in GridProp ptProp, __in Point3D* inPts, __in unsigned int ptSize, __in char lineName, __in size_t outPitch,  __out Point3D* outPts)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;
	//extern __shared__ Point3D curPts[];//blockDim.x*blockDim.y*Point3D
	unsigned int rowID, colID;
	Point3D* curLine;
	while (tID < ptSize)
	{
		if ('y' == lineName || 'Y' == lineName)
		{
			colID = unsigned int((inPts[tID].x3 - ptProp.xbeg) / ptProp.xstep);
			rowID = unsigned int((inPts[tID].y3 - ptProp.ybeg) / ptProp.ystep);
		}
		else if ('x' == lineName || 'X' == lineName)
		{
			colID = unsigned int((inPts[tID].y3 - ptProp.ybeg) / ptProp.ystep);
			rowID = unsigned int((inPts[tID].x3 - ptProp.xbeg) / ptProp.xstep);
		}
		curLine = (Point3D*)((char*)outPts + rowID*outPitch);
		if (!(dIsZero(inPts[tID].x3) && dIsZero(inPts[tID].y3) && dIsZero(inPts[tID].z3)))
		{
			curLine[colID] = inPts[tID];
		}	
		tID += gridDim.x*blockDim.x*blockDim.y;
	}
}

void CountLinePoints(__in Point3D* linePt,__in unsigned int inSize, 
	__out Point3D* outLine,__out Point3Dw* outP, __out unsigned int& lineSize,__out double* perU)
{
	lineSize = 0;
	perU[0] = perU[1] = perU[2] = perU[3] = 0.0;
	double curChordLen = 0.0;
	for (unsigned int ii = 0; ii < inSize; ii++)
	{
		if (!(dIsZero(linePt[ii].x3) && dIsZero(linePt[ii].y3) && dIsZero(linePt[ii].z3)))
		{
			outLine[lineSize] = linePt[ii];
			if (lineSize >= 1)
			{
				curChordLen = sqrt((outLine[lineSize].x3 - outLine[lineSize - 1].x3)*(outLine[lineSize].x3 - outLine[lineSize - 1].x3) +
					(outLine[lineSize].y3 - outLine[lineSize - 1].y3)*(outLine[lineSize].y3 - outLine[lineSize - 1].y3) +
					(outLine[lineSize].z3 - outLine[lineSize - 1].z3)*(outLine[lineSize].z3 - outLine[lineSize - 1].z3));
			}
			perU[3 + lineSize] = perU[3 + lineSize - 1] + curChordLen;
			outP[lineSize] = PointQuat(outLine[lineSize],1.0);
			lineSize++;
		}
	}
	perU[3 + lineSize] = perU[3 + lineSize - 1];
	perU[3 + lineSize + 1] = perU[3 + lineSize + 2] = perU[3 + lineSize];

	for (unsigned int jj = 0; jj < lineSize + 6; jj++)
	{
		perU[jj] /= perU[lineSize + 5];
	}
}

__device__ void FindSpan1(__in int n, __in int p, __in double u, __in double* U, __out int* span)
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

__device__ void BasisFuns1(__in int i, __in double u, __in int p, __in double* U, __in double* left, __in double* right, __out double* N)
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

__device__ void DersBasisFuns1(__in int i, __in int p, __in double u, __in int nd, __in double* U,
	__in double* ndu, __in double* a, __in double* left, __in double* right, __out double* ders)
{
	/* Implementation of Algorithm A2.3 from The NURBS Book by Piegl & Tiller.
	Local arrays:
	ndu[p+1][p+1], to store the basis funtions and knot differences;
	a[2][p+1], to store the two most recently computed rows a(k,j) and a(k-1,j).
	left, right have length of p+1.
	In this function, these local arrays are stored in 1-D array and using following convertation:
	ndu[i][j]=ndu[i*(p+1)+j], and a[i][j]=a[i*2+j].
	The result ders array has the size of (nd+1)*(p+1), therefore,
	ders[i][j]=ders[i*(nd+1)+j].
	*/
	ndu[0] = 1.0;
	double saved, temp;
	for (int ii = 0; ii <= p; ii++)
	{
		for (int jj = 0; jj <= p; jj++)
		{
			ndu[ii*(p + 1) + jj] = 1.0;
		}
		left[ii] = 1.0;
		right[ii] = 1.0;
	}
	for (int j = 1; j <= p; j++)
	{
		left[j] = u - U[i + 1 - j];
		right[j] = U[i + j] - u;
		saved = 0.0;
		for (int r = 0; r < j; r++)
		{
			ndu[j*(p + 1) + r] = right[r + 1] + left[j - r];
			temp = ndu[r*(p + 1) + j - 1] / ndu[j*(p + 1) + r];

			ndu[r*(p + 1) + j] = saved + right[r + 1] * temp;
			saved = left[j - r] * temp;
		}
		ndu[j*(p + 1) + j] = saved;
	}

	for (int ii = 0; ii <= nd; ii++)
	{
		for (int jj = 0; jj <= p; jj++)
		{
			ders[ii*(p + 1) + jj] = 0.0;
		}
	}

	for (int j = 0; j <= p; j++)
	{
		ders[j] = ndu[j*(p + 1) + p];
	}

	/*This section computes the derviatives*/
	int s1, s2, rk, pk, j1, j2, j;
	double d;
	for (int ii = 0; ii < 2; ii++)
	{
		for (int jj = 0; jj <= p; jj++)
		{
			a[ii * (p + 1) + jj] = 1.0;
		}
	}
	for (int r = 0; r <= p; r++)
	{
		s1 = 0, s2 = 1;
		a[0] = 1.0;
		for (int k = 1; k <= nd; k++)
		{
			d = 0.0;
			rk = r - k, pk = p - k;
			if (r >= k)
			{
				a[s2 * (p + 1)] = a[s1 * (p + 1)] / ndu[(pk + 1)*(p + 1) + rk];
				d = a[s2 * (p + 1)] * ndu[rk*(p + 1) + pk];
			}
			if (rk >= -1) j1 = 1;
			else j1 = -rk;

			if (r - 1 <= pk)j2 = k - 1;
			else j2 = p - r;

			for (j = j1; j <= j2; j++)
			{
				a[s2 * (p + 1) + j] = (a[s1 * (p + 1) + j] - a[s1 * (p + 1) + j - 1]) / ndu[(pk + 1)*(p + 1) + rk + j];
				d += a[s2 * (p + 1) + j] * ndu[(rk + 1)*(p + 1) + pk];
			}
			if (r <= pk)
			{
				a[s2 * (p + 1) + k] = -a[s1 * (p + 1) + k - 1] / ndu[(pk + 1)*(p + 1) + r];
				d += a[s2 * (p + 1) + k] * ndu[r*(p + 1) + pk];
			}
			ders[k*(p + 1) + r] = d;
			j = s1, s1 = s2, s2 = j;
		}
	}

	/*Multipy through by the correct factors*/
	double r = p*1.0;
	for (int k = 1; k <= nd; k++)
	{
		for (int j = 0; j <= p; j++)
		{
			ders[k*(p + 1) + j] *= r;
		}
		r *= (p - k);
	}
}

__device__ void CurvePoint3D(__in int span, __in int p, __in int n, __in double* N, __in Point3Dw* ctrPt, __out Point3D* resPt)
{
	double denom = 0.0;
	Point3Dw tmp;
	for (int ii = 0; ii <= p; ii++)
	{
		if (span - p + ii >= 0 && span - p + ii <= n)
		{
			tmp += (ctrPt[span - p + ii].HomoCoordPtToPt3Dw()) * N[ii];
			denom += ctrPt[span - p + ii].w*N[ii];
		}
	}
	if (abs(denom) > 1e-6)
	{
		tmp /= denom;
		tmp.GetPoint3D(*resPt);
	}
}

__device__ void CurvePointCurvature3D(__in int p, __in int m, __in double* U, __in int i, __in double u, __in double* N,
	__in Point3Dw* ctrPts, __in double* left, __in double* right,
	__in double* a, __in double* ders2, __in Point3Dw* ders,
	__out double* curvature, __out Point3Dw* norms, __out Point3Dw* tans, __out Point3Dw* binorms)
{
	int derD = 2, du = min(derD, p);
	Point3Dw curPt, der1Pt, der2Pt;
	DersBasisFuns1(i, p, u, derD, U, N, a, left, right, ders2);

	for (int k = p + 1; k <= derD; k++)
	{
		ders[k] = Point3Dw();
	}
	for (int k = 0; k <= du; k++)
	{
		for (int j = 0; j <= p; j++)
		{
			if (NULL != ctrPts)
			{
				curPt += ders2[k*(p + 1) + j] * ctrPts[i - p + j];
			}
		}
		ders[k] = curPt;
	}
	der1Pt = ders[1];
	der2Pt = ders[2];
	*curvature = (der1Pt.Cross(der2Pt)).Length() / der1Pt.Length() / der1Pt.Length() / der1Pt.Length();

	Point3Dw tanV, normV, binormV;
	tanV = der1Pt;
	tanV /= tanV.Length();

	binormV = der1Pt.Cross(der2Pt);
	binormV /= binormV.Length();

	normV = tanV.Cross(binormV);
	normV /= normV.Length();

	if (NULL != norms)
	{
		*norms = normV;
	}

	if (NULL != tans)
	{
		*tans = tanV;
	}

	if (NULL != binorms)
	{
		*binorms = binormV;
	}
}

__global__ void GetControlPointsFromDataPointP3D(__in Point3D* dataPt, __in unsigned int dataSize,
	__in PointQuat* P, __in double* A, __in double* B, __in double* C, __in double* _C, __in PointQuat* E, __in PointQuat* _E,
	__in double* U, __out Point3Dw* ctrPt)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;
		//A,B,C,_C size 1*DS, E,_E size 4*DS
		/* A,B,C are vectors which contain values of tri-diagonal matrix with the following form:
		|a1	b1	c1							|
		|a2	b2	c2							|
		|	a3	b3	c3						|
		|		...	...	...					|
		|			a_DS-1_	b_DS-1_	c_DS-1_	|
		|			a_DS_	b_DS_	c_DS_	|
		Therefore, A={a1,a2,...,a_DS_}, B={b1,b2,...,b_DS_} and C={c1,c2,...c_DS_}.
		In algorithm, the indices of vectors are start from 0, so A[i]=a_i+1_, for i=0,1,...DS-1
		*/
	unsigned int tmpID;

	if (tID < dataSize - 2)//dataSize - 1 = n
	{
		//i=2,3,...,dataSize-1, tID=0,1,...,dataSize-3
		//A[i]=(u_i+3-u_i+2)^2/(u_i+3-u_i)
		tmpID = tID + 2;
		A[tmpID - 1] = (U[tmpID + 3] - U[tmpID + 2])*(U[tmpID + 3] - U[tmpID + 2]) / (U[tmpID + 3] - U[tmpID]);
		//B[j]=(u_i+3-u_i+2)*(u_i+2-u_i)/(u_i+3-u_i)+(u_i+2-u_i+1)*(u_i+4-u_i+2)/(u_i+4-u_i+1)
		B[tmpID - 1] = (U[tmpID + 3] - U[tmpID + 2])*(U[tmpID + 2] - U[tmpID]) / (U[tmpID + 3] - U[tmpID]) +
			(U[tmpID + 2] - U[tmpID + 1])*(U[tmpID + 4] - U[tmpID + 2]) / (U[tmpID + 4] - U[tmpID + 1]);
		//C[j]=(u_i+2-u_i+1)^2/(u_i+4-u_i+1)
		C[tmpID - 1] = (U[tmpID + 2] - U[tmpID + 1])*(U[tmpID + 2] - U[tmpID + 1]) / (U[tmpID + 4] - U[tmpID + 1]);
		//E[j]=(u_i+3-u_i+1)*P_i-1
		E[tmpID - 1] = P[tmpID - 1] * (U[tmpID + 3] - U[tmpID + 1]);
	}

	__syncthreads();
	__shared__ PointQuat curD, pre1D, pre2D;
	__shared__ double __B0;

	if (0 == tID)
	{//Free end boundary condition
		A[0] = 2 - (U[4] - U[3])*(U[5] - U[4]) / (U[5] - U[3]) / (U[5] - U[3]);
		B[0] = (U[4] - U[3]) / (U[5] - U[3])*((U[5] - U[4]) / (U[5] - U[3]) - (U[4] - U[3]) / (U[6] - U[3]));
		C[0] = (U[4] - U[3])*(U[4] - U[3]) / (U[5] - U[3]) / (U[6] - U[3]);
		tmpID = dataSize - 1;
		A[dataSize - 1] = (U[dataSize + 2] - U[dataSize + 1])*(U[dataSize + 2] - U[dataSize + 1]) /
			(U[dataSize + 2] - U[dataSize]) / (U[dataSize + 2] - U[dataSize - 1]);
		B[dataSize - 1] = (U[dataSize + 2] - U[dataSize + 1]) / (U[dataSize + 2] - U[dataSize]) *
			((U[dataSize + 2] - U[dataSize + 1]) / (U[dataSize + 2] - U[dataSize - 1]) - (U[dataSize + 1] - U[dataSize]) / (U[dataSize + 2] - U[dataSize]));
		C[dataSize - 1] = (U[dataSize + 1] - U[dataSize])*(U[dataSize + 2] - U[dataSize + 1]) /
			(U[dataSize + 2] - U[dataSize]) / (U[dataSize + 2] - U[dataSize]) - 2;
		E[0] = P[0] + P[1];
		E[tmpID] = P[tmpID] * (-1.0) - P[tmpID - 1];
		//End of Free end boundary condition

		//Modified Tri-diagonal matrix
		__B0 = B[0] / A[0];
		_C[0] = C[0] / A[0];
		_C[1] = (C[1] * A[0] - A[1] * C[0]) / (B[1] * A[0] - A[1] * B[0]);
		_E[0] /= A[0];
		_E[1] = (E[1] - E[0] * A[1]) / (B[1] * A[0] - A[1] * B[0]);
		for (int ii = 2; ii < dataSize - 1; ii++)
		{
			_C[ii] = C[ii] / (B[ii] - A[ii] * _C[ii - 1]);
			_E[ii] = (E[ii] - _E[ii - 1] * A[ii]) / (B[ii] - A[ii] * _C[ii - 1]);
		}
		tmpID = dataSize - 1;
		_C[tmpID] = C[tmpID] / (B[tmpID] - A[tmpID] * _C[tmpID - 1]) - _C[tmpID];
		_E[tmpID] = (E[tmpID] - _E[tmpID - 2] * A[tmpID]) / (B[tmpID] - A[tmpID] * _C[tmpID - 2]) - _E[tmpID - 1];

		//Chasing method to solve linear system of tri-diagonal matrix
		ctrPt[dataSize + 1] = Point3Dw(dataPt[dataSize - 1], 1.0);

		pre1D = _E[dataSize - 1] / _C[dataSize - 1];

		ctrPt[dataSize] = pre1D.HomoCoordPtToPt3Dw();

		for (unsigned int ii = dataSize - 2; ii >= 1; ii--)
		{
			curD = _E[ii] - pre1D * _C[ii];
			ctrPt[ii + 1] = Point3Dw(curD.x / curD.w, curD.y / curD.w, curD.z / curD.w, curD.w);

			if (ii == 1)
			{
				pre2D = pre1D;
			}
			pre1D = curD;
		}
		curD = _E[1] - pre1D *__B0 - pre2D*_C[0];
		ctrPt[1] = curD.HomoCoordPtToPt3Dw();
		ctrPt[0] = Point3Dw(dataPt[0], 1.0);
	}
}

__global__ void GetCurvePointAndCurvature3D(__in Point3Dw* ctrPts, __in int ctrSize, __in size_t pcSize, __in double* inU,
		__inout Point3D* cvPts, __inout double* curvatures, __out Point3D* ptTans, __out Point3D* ptNorms)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;
	int p = 3,             //degree of basis function
		n = ctrSize - 1,  //(n+1) control points,id=0,1,...,n.
		m = n + p + 1;     //Size of knot vector(m+1),id=0,1,2,...,m.
	extern __shared__ double U[];//Knot vector

	Point3D curCurve;

	if (tID < pcSize)
	{
		int span = -1;// , prespan = -1;
		double* N = (double*)malloc((p + 1) * sizeof(double)),
			*left = (double*)malloc((p + 1) * sizeof(double)),
			*right = (double*)malloc((p + 1) * sizeof(double));
		double u = tID*1.0 / (pcSize*1.0);
		FindSpan1(n, p, u, inU, &span);
		BasisFuns1(span, u, p, inU, left, right, N);
		CurvePoint3D(span, p, n, N, ctrPts, &curCurve);
		cvPts[tID] = curCurve;
		cvPts[tID].id = tID;
		
		double* Ndu = (double*)malloc((p + 1)*(p + 1) * sizeof(double));
		double* a = (double*)malloc(2 * (p + 1) * sizeof(double)),
			*ders2 = (double*)malloc(3 * (p + 1) * sizeof(double));
		double curCvt;
		Point3Dw* ders = (Point3Dw*)malloc(3 * sizeof(Point3Dw)),
			*norms = (Point3Dw*)malloc(sizeof(Point3Dw)),
			*tans = (Point3Dw*)malloc(sizeof(Point3Dw)),
			*binorms = (Point3Dw*)malloc(sizeof(Point3Dw));
		CurvePointCurvature3D(p, m, inU, span, u, Ndu, ctrPts, left, right, a, ders2, ders, &curCvt, norms, tans, binorms);

		curvatures[tID] = curCvt;

		if (NULL != ptTans)
		{
			ptTans[tID] = tans->HomoCoordPtToPt3D();
			ptTans[tID].id = tID;
		}

		if (NULL != norms)
		{
			ptNorms[tID] = norms->HomoCoordPtToPt3D();
			ptNorms[tID].id = tID;
		}

		free(a);
		a = NULL;
		free(ders2);
		ders2 = NULL;
		free(Ndu);
		Ndu = NULL;
		free(N);
		N = NULL;
		free(left);
		left = NULL;
		free(right);
		right = NULL;
		free(ders);
		ders = NULL;
		free(norms);
		norms = NULL;
		free(tans);
		tans = NULL;
		free(binorms);
		binorms = NULL;
	}
}

__global__ void ClearCurvePoints(__in GridProp ptProp, __in Point3D* cvPts,__in unsigned int cvSize, __out Point3D* outcvPts)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;

	if (tID < cvSize)
	{
		if (ptProp.xbeg < cvPts[tID].x3 && cvPts[tID].x3 < ptProp.xend &&
			ptProp.ybeg < cvPts[tID].y3 && cvPts[tID].y3 < ptProp.yend)
		{
			outcvPts[tID] = cvPts[tID];
		}
	}

	/*unsigned int validSize = 0;
	for (unsigned int ii = 0; ii < cvSize; ii++)
	{
		if (ptProp.xbeg < cvPts[ii].x3 && cvPts[ii].x3 < ptProp.xend &&
			ptProp.ybeg < cvPts[ii].y3 && cvPts[ii].y3 < ptProp.yend)
		{
			outcvPts[validSize] = cvPts[ii];
		}
	}*/
}

__global__ void GetDataPointCurvature3D(__in GridProp ptProp, __in char lineName, __in Point3D* cvPts,__in Point3D* cvNorms, __in Point3D* dtPts, __in unsigned int cvSize, __in double* cvCurvs, __in unsigned int dtSize, __out double* dtCurvs, __out Point3D* dtNorms)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;

	if (tID < dtSize)
	{
		Point3D curPt = dtPts[tID];
		unsigned int upID = cvSize - 2, lowID = 1;
		unsigned int invUpID = 0, invLowID = 0;//Invalid curve points boundary IDs.
		unsigned int curID;
		float ptVal = 0.0f, cvVal = 0.0f, invLowVal = 0.0f, invUpVal = 0.0f;
		float valUpB = 0.0f, valLowB = 0.0f;
		switch (lineName)
		{
		case 'x':
		case 'X':
			ptVal = curPt.y3;
			valUpB = ptProp.yend;
			valLowB = ptProp.ybeg;
			break;
		case 'y':
		case 'Y':
			ptVal = curPt.x3;
			valUpB = ptProp.xend;
			valLowB = ptProp.xbeg;
			break;
		}

		while (upID - lowID >= 2)
		{
			curID = (upID + lowID) / 2;
			switch (lineName)
			{
			case 'x':
			case 'X':
				cvVal = cvPts[curID].y3;
				break;
			case 'y':
			case 'Y':
				cvVal = cvPts[curID].x3;
				break;
			}
			if (cvVal<valLowB || cvVal>valUpB)
			{
				invLowID = curID;
				invLowVal = cvVal;
				while (invLowVal<valLowB || invLowVal>valUpB)
				{
					invLowID--;
					if (lineName == 'x' || lineName == 'X')
					{
						invLowVal = cvPts[invLowID].y3;
					}
					else if (lineName == 'y' || lineName == 'Y')
					{
						invLowVal = cvPts[invLowID].x3;
					}
					if (invLowID == 0)
					{
						break;
					}
				}

				invUpID = curID;
				invUpVal = cvVal;
				while (invUpVal<valLowB || invUpVal>valUpB)
				{
					invUpID++;
					if (lineName == 'x' || lineName == 'X')
					{
						invUpVal = cvPts[invUpID].y3;
					}
					else
					{
						invUpVal = cvPts[invUpID].x3;
					}

					if (invUpID == cvSize - 2)
					{
						break;
					}
				}
				if (ptVal > invUpVal)
				{
					curID = invUpID;
					cvVal = invUpVal;
				}

				if (ptVal < invLowVal)
				{
					curID = invLowID;
					cvVal = invLowVal;
				}
			}

			if (ptVal < cvVal)
			{
				upID = curID;
			}
			else
			{
				lowID = curID;
			}
		}
		float lowW = P3DLen(P3DSub(curPt, cvPts[lowID])) / (P3DLen(P3DSub(curPt, cvPts[lowID])) + P3DLen(P3DSub(cvPts[upID], curPt)));
		dtCurvs[tID] = cvCurvs[lowID] * lowW + cvCurvs[upID] * (1 - lowW);
		dtNorms[tID] = P3DSum(P3DMult(cvNorms[lowID], lowW), P3DMult(cvNorms[upID], 1 - lowW));
		dtNorms[tID] = P3DDev(dtNorms[tID], P3DLen(dtNorms[tID]));
	}
} 

__global__ void GetHighCurvaturePt(__in Point3D* dtPts, __in double* dtCurv, __in Point3D* dtNorms, __in unsigned int dtSize, 
	__in float cvThrs,__in float dgThrs, __in unsigned int nghbs, __out int * resID)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;

	if (tID < dtSize)
	{
		resID[tID] = -1;
		unsigned int begID, endID;
		if (nghbs / 2 > tID)
		{
			begID = 0;
			endID = nghbs;
		}
		else if (dtSize - nghbs / 2 < tID)
		{
			begID = dtSize - nghbs;
			endID = dtSize - 1;
		}
		else
		{
			begID = tID - nghbs / 2;
			endID = begID + nghbs;
		}
		float avgCv = 0.0, normDot = 0.0, denom = 0.0, wt;
		Point3D AvgNorm;
		AvgNorm.x3 = 0.0f;
		AvgNorm.y3 = 0.0f;
		AvgNorm.z3 = 0.0f;
		for (unsigned int ii = begID + 1; ii < endID; ii++)
		{
			if (ii < tID)
			{
				wt = (float)(nghbs / 2 - (tID - ii));
			}
			else
			{
				wt = (float)(nghbs / 2 - (ii - tID));
			}
			if (ii != tID)
			{
				normDot += P3DDot(dtNorms[ii], dtNorms[tID])*wt;
				AvgNorm = P3DSum(P3DMult(dtNorms[ii], wt), AvgNorm);
				avgCv += dtCurv[ii] * wt;
				denom += wt;
			}
		}
		avgCv /= denom;
		normDot /= (denom * P3DLen(AvgNorm) * P3DLen(dtNorms[tID]));
		double preCos = 1.0, postCos = 1.0;
		float preCv = dtCurv[tID] + 1.0, postCv = dtCurv[tID] + 1.0;

		if (tID >= 1)
		{
			preCos = P3DDot(dtNorms[tID - 1], dtNorms[tID]) / P3DLen(dtNorms[tID - 1]) / P3DLen(dtNorms[tID]);
			preCv = dtCurv[tID - 1];
		}
		if (tID <= dtSize)
		{
			postCos = P3DDot(dtNorms[tID], dtNorms[tID + 1]) / P3DLen(dtNorms[tID]) / P3DLen(dtNorms[tID + 1]);
			postCv = dtCurv[tID + 1];
		}
		if ((preCv*cvThrs < dtCurv[tID] || postCv*cvThrs < dtCurv[tID]) ||
			(abs(preCos) < cos(dgThrs / 180 * 3.1415926535) || abs(postCos) < cos(dgThrs / 180 * 3.14159269535)))
		{
			resID[tID] = dtPts[tID].id;
		}
		else
		{
			resID[tID] = -1;
		}
	}
}


const char* GetCUDAErrorInfo(cudaError_t inErr, char* refInfo)
{
	char *totalInfo, *tmpstr, retstr[1000];
	totalInfo = (char*)malloc(1000 * sizeof(char));
	tmpstr = (char*)malloc(1000 * sizeof(char));
	sprintf(totalInfo, "Error position: %s \r\n", refInfo);
	sprintf(tmpstr, "Error information: %s \r\n", cudaGetErrorString(inErr));
	totalInfo = strcat(totalInfo, tmpstr);
	strcpy(retstr, totalInfo);
	free(totalInfo);
	totalInfo = NULL;
	free(tmpstr);
	tmpstr = NULL;
	return retstr;
}

const char* NURBSCurveSearchGPU(GridProp ptProp, Point3D* inPts, unsigned int ptSize, char lineName,
	float cvThrs, unsigned int cvSize, float dgThrs, unsigned int nghbs, int* outIDs, Point3Dw* outCtrs, Point3D* outCvPts)
{
	unsigned int gridX = 1, gridY = 1;
	GetGridFactor((unsigned int)ptSize, (unsigned int)THREAD_SIZE*THREAD_SIZE, gridX, gridY);
	dim3 blocks(THREAD_SIZE, THREAD_SIZE);
	cudaDeviceProp curProp;
	cudaERRORHANDEL(cudaGetDeviceProperties(&curProp, 0), "DeviceProp");

	int grids = curProp.multiProcessorCount * 100;
	if ((int)gridX*gridY < grids)
	{
		grids = (int)gridX*gridY;
	}

	size_t linePitch;
	Point3D* d_linePt;
	size_t width = (size_t)(ptProp.xend - ptProp.xbeg) / ptProp.xstep + 1,
		height = (size_t)(ptProp.yend - ptProp.ybeg) / ptProp.ystep + 1;
	if ('y' == lineName || 'Y' == lineName)
	{
		width = (size_t)(ptProp.xend - ptProp.xbeg) / ptProp.xstep + 1;
		height = (size_t)(ptProp.yend - ptProp.ybeg) / ptProp.ystep + 1;
	}
	else if ('x' == lineName || 'X' == lineName)
	{
		width = (size_t)(ptProp.yend - ptProp.ybeg) / ptProp.ystep + 1;
		height = (size_t)(ptProp.xend - ptProp.xbeg) / ptProp.xstep + 1;
	}
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_linePt, &linePitch, width * sizeof(Point3D), height), "MallocPitch d_linePt");
	Point3D* d_inPts;
	cudaERRORHANDEL(cudaMalloc((void**)&d_inPts, ptSize * sizeof(Point3D)), "Malloc d_inPts");
	cudaERRORHANDEL(cudaMemcpy(d_inPts, inPts, ptSize * sizeof(Point3D), cudaMemcpyHostToDevice), "Copy d_inPts");

	GetOneLineData <<<grids, blocks >>> (ptProp, d_inPts, ptSize, lineName, linePitch, d_linePt);
	Point3D* h_inPts;
	h_inPts = (Point3D*)malloc(width*height * sizeof(Point3D));
	cudaERRORHANDEL(cudaMemcpy2D(h_inPts, width * sizeof(Point3D), d_linePt, linePitch, width * sizeof(Point3D), height, cudaMemcpyDeviceToHost), "Copy2D d_linePt");

	Point3D* h_oneLine, *h_useData, *d_useData, *h_bckuseD;
	unsigned int dataSize;
	h_useData = (Point3D*)malloc(width * sizeof(Point3D));
	double *d_A, *d_B, *d_C, *d__C, *d_U, *h_U;
	h_U = (double*)malloc((width + 6) * sizeof(double));
	Point3Dw* d_P, *d_E, *d__E, *d_ctrPts, *h_ctrPts,*d_totalCtr;
	PointQuat *h_P;
	h_P = (PointQuat*)malloc(width * sizeof(PointQuat));

	Point3D* d_cvPts, *d_cvTans, *d_cvNorms, *d_cvTotalPt, *d_dtNorms, *d_validCvPts;
	Point3D* h_cvPts, *h_cvTans, *h_cvNorms, *h_dtNorms, *h_validCvPts;
	double* d_cvCurvature, *d_dtCurvature;
	double* h_cvCurvature, *h_dtCurvature;
	int* d_resID, *h_resID, *d_totalID;
	unsigned int cvPoints = cvSize;
	cudaERRORHANDEL(cudaMalloc((void**)&d_cvPts, cvPoints * sizeof(Point3D)), "Malloc d_cvPts");
	cudaERRORHANDEL(cudaMalloc((void**)&d_validCvPts, cvPoints * sizeof(Point3D)), "Malloc d_validCvPts");
	cudaERRORHANDEL(cudaMalloc((void**)&d_cvTotalPt, cvPoints*height * sizeof(Point3D)), "Malloc d_cvTotalPt");
	cudaERRORHANDEL(cudaMalloc((void**)&d_cvTans, cvPoints * sizeof(Point3D)), "Malloc d_cvTans");
	cudaERRORHANDEL(cudaMalloc((void**)&d_cvNorms, cvPoints * sizeof(Point3D)), "Malloc d_cvNorms");
	cudaERRORHANDEL(cudaMalloc((void**)&d_cvCurvature, cvPoints * sizeof(double)), "Malloc d_cvCurvature");

	h_cvPts = (Point3D*)malloc(cvPoints * sizeof(Point3D));
	h_validCvPts = (Point3D*)malloc(cvPoints * sizeof(Point3D));
	h_cvTans = (Point3D*)malloc(cvPoints * sizeof(Point3D));
	h_cvNorms = (Point3D*)malloc(cvPoints * sizeof(Point3D));
	h_cvCurvature = (double*)malloc(cvPoints * sizeof(double));

	dim3 cvBlocks(16, 16), cvGrids;
	GetGridFactor((unsigned int)cvPoints, (unsigned int)cvBlocks.x*cvBlocks.y, cvGrids.x, cvGrids.y);

	dim3 dtBlocks(16, 16), dtGrids;

	cudaERRORHANDEL(cudaMalloc((void**)&d_useData, width * sizeof(Point3D)), "Malloc d_useData");
	cudaERRORHANDEL(cudaMalloc((void**)&d_ctrPts, (width + 2) * sizeof(Point3Dw)), "Malloc d_ctrPts");
	cudaERRORHANDEL(cudaMalloc((void**)&d_totalCtr, (width + 2) * height * sizeof(Point3Dw)), "Malloc d_totalCtr");
	cudaERRORHANDEL(cudaMalloc((void**)&d_P, width * sizeof(PointQuat)), "Malloc d_P");
	cudaERRORHANDEL(cudaMalloc((void**)&d_A, width * sizeof(double)), "Malloc d_A");
	cudaERRORHANDEL(cudaMalloc((void**)&d_B, width * sizeof(double)), "Malloc d_B");
	cudaERRORHANDEL(cudaMalloc((void**)&d_C, width * sizeof(double)), "Malloc d_C");
	cudaERRORHANDEL(cudaMalloc((void**)&d_E, width * sizeof(PointQuat)), "Malloc d_E");
	cudaERRORHANDEL(cudaMalloc((void**)&d__C, width * sizeof(double)), "Malloc d__C");
	cudaERRORHANDEL(cudaMalloc((void**)&d__E, width * sizeof(PointQuat)), "Malloc d__E");
	cudaERRORHANDEL(cudaMalloc((void**)&d_U, (width + 6) * sizeof(double)), "Malloc d_U");
	cudaERRORHANDEL(cudaMalloc((void**)&d_dtCurvature, width * sizeof(double)), "Malloc d_dtCurvature");
	cudaERRORHANDEL(cudaMalloc((void**)&d_dtNorms, width * sizeof(Point3D)), "Malloc d_dtNorms");
	cudaERRORHANDEL(cudaMalloc((void**)&d_resID, width * sizeof(int)), "Malloc d_resID");
	cudaERRORHANDEL(cudaMalloc((void**)&d_totalID, width*height * sizeof(int)), "Malloc d_totalID");
	cudaERRORHANDEL(cudaMemset(d_totalID, -1, width*height * sizeof(int)), "Memset d_totalID");
	h_bckuseD = (Point3D*)malloc(width * sizeof(Point3D));
	h_dtCurvature = (double*)malloc(width * sizeof(double));
	h_dtNorms = (Point3D*)malloc(width * sizeof(Point3D));
	h_resID = (int*)malloc(width * sizeof(int));
	h_ctrPts = (Point3Dw*)malloc((width + 2) * sizeof(Point3Dw));
	for (unsigned int ii = 0; ii < height; ii++)
	{
		h_oneLine = &h_inPts[ii*width];
		//memset(h_useData, 0, width * sizeof(Point3D));
		//memset(h_P, 0, width * sizeof(PointQuat));
		//memset(h_U, 0, (width + 6) * sizeof(double));

		CountLinePoints(h_oneLine, width, h_useData, h_P, dataSize, h_U);
		
		if (dataSize <= 5)
		{
			continue;
		}
		GetGridFactor((unsigned int)dataSize, (unsigned int)dtBlocks.x*dtBlocks.y, dtGrids.x, dtGrids.y);
		/*cudaERRORHANDEL(cudaMalloc((void**)&d_useData, dataSize * sizeof(Point3D)), "Malloc d_useData");
		cudaERRORHANDEL(cudaMalloc((void**)&d_ctrPts, (dataSize + 2) * sizeof(Point3Dw)), "Malloc d_ctrPts");
		cudaERRORHANDEL(cudaMalloc((void**)&d_P, dataSize * sizeof(PointQuat)), "Malloc d_P");
		cudaERRORHANDEL(cudaMalloc((void**)&d_A, dataSize * sizeof(double)), "Malloc d_A");
		cudaERRORHANDEL(cudaMalloc((void**)&d_B, dataSize * sizeof(double)), "Malloc d_B");
		cudaERRORHANDEL(cudaMalloc((void**)&d_C, dataSize * sizeof(double)), "Malloc d_C");
		cudaERRORHANDEL(cudaMalloc((void**)&d_E, dataSize * sizeof(PointQuat)), "Malloc d_E");
		cudaERRORHANDEL(cudaMalloc((void**)&d__C, dataSize * sizeof(double)), "Malloc d__C");
		cudaERRORHANDEL(cudaMalloc((void**)&d__E, dataSize * sizeof(PointQuat)), "Malloc d__E");
		cudaERRORHANDEL(cudaMalloc((void**)&d_U, (dataSize + 6) * sizeof(double)), "Malloc d_U");
		cudaERRORHANDEL(cudaMalloc((void**)&d_dtCurvature, dataSize * sizeof(double)), "Malloc d_dtCurvature");
		cudaERRORHANDEL(cudaMalloc((void**)&d_resID, dataSize * sizeof(int)), "Malloc d_resID");
		h_bckuseD = (Point3D*)malloc(dataSize * sizeof(Point3D));
		h_dtCurvature = (double*)malloc(dataSize * sizeof(double));
		h_resID = (int*)malloc(dataSize * sizeof(int));
		h_ctrPts = (Point3Dw*)malloc((dataSize + 2) * sizeof(Point3Dw));*/

		cudaERRORHANDEL(cudaMemcpy(d_useData, h_useData, dataSize * sizeof(Point3D), cudaMemcpyHostToDevice), "Copy d_useData");
		cudaERRORHANDEL(cudaMemcpy(d_P, h_P, dataSize * sizeof(PointQuat), cudaMemcpyHostToDevice), "Copy d_P");
		cudaERRORHANDEL(cudaMemcpy(d_U, h_U, (dataSize + 6) * sizeof(double), cudaMemcpyHostToDevice), "Copy d_U");


		GetControlPointsFromDataPointP3D<<<dtGrids, dtBlocks >>>(d_useData, dataSize,
			d_P, d_A, d_B, d_C, d__C, d_E, d__E, d_U, d_ctrPts);
		//cudaERRORHANDEL(cudaDeviceSynchronize(), "GetControlPointsFromDataPointP3D");
		cudaERRORHANDEL(cudaGetLastError(), "GetControlPointsFromDataPointP3D");
		cudaERRORHANDEL(cudaMemcpy(h_bckuseD, d_useData, dataSize * sizeof(Point3D), cudaMemcpyDeviceToHost), "GetControlPointsFromDataPointP3D");
		cudaERRORHANDEL(cudaMemcpy(h_ctrPts, d_ctrPts, (dataSize + 2) * sizeof(Point3Dw), cudaMemcpyDeviceToHost), "GetControlPointsFromDataPointP3D");

		GetCurvePointAndCurvature3D<<<cvGrids,cvBlocks,(dataSize+6)*sizeof(double)>>>(d_ctrPts, dataSize + 2, cvPoints, d_U, d_cvPts, d_cvCurvature, d_cvTans, d_cvNorms);
		//cudaERRORHANDEL(cudaDeviceSynchronize(), "GetCurvePointAndCurvature3D");
		cudaERRORHANDEL(cudaGetLastError(), "GetCurvePointAndCurvature3D");
		cudaERRORHANDEL(cudaMemcpy(d_cvTotalPt+ii*cvPoints, d_cvPts, cvPoints * sizeof(Point3D), cudaMemcpyDeviceToDevice), "Memcpy d_cvTotalPt");

		cudaERRORHANDEL(cudaMemcpy(h_cvPts, d_cvPts, cvPoints * sizeof(Point3D), cudaMemcpyDeviceToHost), "Copy h_cvPts");
		//cudaERRORHANDEL(cudaMemcpy(h_cvTans, d_cvTans, cvPoints * sizeof(Point3D), cudaMemcpyDeviceToHost), "GetCurvePointAndCurvature3D");
		//cudaERRORHANDEL(cudaMemcpy(h_cvNorms, d_cvNorms, cvPoints * sizeof(Point3D), cudaMemcpyDeviceToHost), "GetCurvePointAndCurvature3D");
		cudaERRORHANDEL(cudaMemcpy(h_cvCurvature, d_cvCurvature, cvPoints * sizeof(double), cudaMemcpyDeviceToHost), "Copy h_cvCurvature");

		ClearCurvePoints <<<cvGrids, cvBlocks>>> (ptProp, d_cvPts, cvSize, d_validCvPts);
		cudaERRORHANDEL(cudaMemcpy(h_validCvPts, d_validCvPts, cvPoints * sizeof(Point3D), cudaMemcpyDeviceToHost), "Copy h_validCvPts");


		GetDataPointCurvature3D<<<dtGrids,dtBlocks>>>(ptProp, lineName, d_validCvPts,d_cvNorms, d_useData, cvPoints, d_cvCurvature,dataSize, d_dtCurvature, d_dtNorms);
		//cudaERRORHANDEL(cudaDeviceSynchronize(), "GetDataPointCurvature3D");
		cudaERRORHANDEL(cudaGetLastError(), "GetDataPointCurvature3D");
		cudaERRORHANDEL(cudaMemcpy(h_dtCurvature, d_dtCurvature, dataSize * sizeof(double), cudaMemcpyDeviceToHost), "Copy h_dtCurvature");
		cudaERRORHANDEL(cudaMemcpy(h_dtNorms, d_dtNorms, dataSize * sizeof(Point3D), cudaMemcpyDeviceToHost), "Copy h_dtNorms");

		GetHighCurvaturePt << <dtGrids, dtBlocks >> > (d_useData, d_dtCurvature, d_dtNorms, dataSize, cvThrs,dgThrs, nghbs, d_resID);
		//cudaERRORHANDEL(cudaDeviceSynchronize(), "GetHighCurvaturePt");
		cudaERRORHANDEL(cudaGetLastError(), "GetHighCurvaturePt");
		memset(h_resID, 0, width * sizeof(int));
		cudaERRORHANDEL(cudaMemcpy(h_resID, d_resID, dataSize * sizeof(int), cudaMemcpyDeviceToHost), "Copy h_resID");
		cudaERRORHANDEL(cudaMemcpy(d_totalID+width*ii, d_resID, width * sizeof(int), cudaMemcpyDeviceToDevice), "Copy d_totalID+width*ii");
		cudaERRORHANDEL(cudaMemcpy(d_totalCtr + (width+2)*ii, d_ctrPts, (width+2) * sizeof(Point3Dw), cudaMemcpyDeviceToDevice), "Copy d_totalCtr+(width+2)*ii");

		//memcpy(outIDs + ii*width, h_resID, width);
		//FreeCudaPtrs(12, &d_useData, &d_P, &d_A, &d_B, &d_C, &d__C, &d_E, &d__E, &d_U, &d_ctrPts, &d_dtCurvature,&d_resID);
		//FreeCPtrs(4, &h_bckuseD, &h_ctrPts, &h_dtCurvature, &h_resID);
	}
	cudaERRORHANDEL(cudaMemcpy(outIDs, d_totalID, width*height * sizeof(int), cudaMemcpyDeviceToHost), "Total Res");
	cudaERRORHANDEL(cudaMemcpy(outCtrs, d_totalCtr, (width+2)*height * sizeof(Point3Dw), cudaMemcpyDeviceToHost), "Total Res");
	cudaERRORHANDEL(cudaMemcpy(outCvPts, d_cvTotalPt, cvPoints*height * sizeof(Point3D), cudaMemcpyDeviceToHost), "Total Res");

	FreeCudaPtrs(22, &d_useData, &d_inPts, &d_linePt,
		&d_P, &d_A, &d_B, &d_C, &d__C, &d_E, &d__E, &d_U,
		&d_ctrPts, &d_cvPts, &d_cvTans, &d_cvNorms, &d_cvCurvature,
		&d_dtCurvature, &d_resID, &d_totalID, &d_totalCtr, &d_cvTotalPt,
		&d_dtNorms);
	FreeCPtrs(12, &h_inPts, &h_useData, &h_U, &h_P, &h_ctrPts,
		&h_bckuseD, &h_cvPts, &h_cvTans, &h_cvNorms, &h_cvCurvature, &h_dtCurvature,
		&h_dtNorms);
	return "Success";
}

__device__ void MergeStream(double* xs, double* ys, double* bx, double* by, int low, int mid, int high)
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

__device__ void MergeSortAlongStream(double* xs, double* ys, char axisName, double* tmpx, double* tmpy, int len)
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

	int k = 1, i = 0;
	while (k < len)
	{
		i = 0;
		while (i + 2 * k <= len)
		{
			MergeStream(xs, ys, tmpx, tmpy, i, i + k, i + 2 * k);
			i += 2 * k;
		}

		if (i + k <= len)
			MergeStream(xs, ys, tmpx, tmpy, i, i + k, len);

		k *= 2;
	}
}

__global__ void ConvertF2IMatStream(float* fVals, int fLen, size_t uiPitch, size_t width, size_t uiMax, size_t *uiVals)
{
	int bID = blockIdx.x + blockIdx.y*gridDim.x;
	int tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;//Thread ID, as iteration ID.
	size_t* pLine = uiVals;
	while (tID < fLen)
	{
		pLine = (size_t*)((char*)uiVals + (tID / width)*uiPitch);
		if (fVals[tID] < 1e-6)
		{
			pLine[tID%width] = 0;
		}
		else
		{
			pLine[tID%width] = size_t((fVals[tID] - 1e-6) / (1 - 1e-6)*uiMax);
		}
		tID += blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	}
}

__global__ void GetHypoXYAtOnceStream(double* xs, double* ys, size_t* hypoID2D, size_t idPitch, size_t width, size_t height, double* hypoxs, size_t xPitch, double* hypoys, size_t yPitch)
{
	unsigned int bID = blockIdx.y*gridDim.x + blockIdx.x,
		tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	unsigned int curRow, curCol;
	while (tID < width*height)
	{
		curRow = tID / width;
		curCol = tID % width;
		double* xsLine = (double*)((char*)hypoxs + curRow*xPitch),
			*ysLine = (double*)((char*)hypoys + curRow*yPitch);
		size_t* idLine = (size_t*)((char*)hypoID2D + curRow*idPitch);
		xsLine[curCol] = xs[idLine[curCol]];
		ysLine[curCol] = ys[idLine[curCol]];
		tID += blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	}
}

__global__ void SortHypoXYStream(CtrPtBound inBd, size_t width, size_t height, double* hypoxs, size_t xPitch, double* hypoys, size_t yPitch)
{
	unsigned int bID = blockIdx.y*gridDim.x + blockIdx.x,
		tID = bID*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	double* xs = (double*)((char*)hypoxs + xPitch*tID), *ys = (double*)((char*)hypoys + yPitch*tID);
	//QSortAlong(xs, ys, 'x', 0, width - 1);

	double* tmpx = (double*)malloc(width * sizeof(double)), *tmpy = (double*)malloc(width * sizeof(double));

	while (tID < height)
	{
		xs = (double*)((char*)hypoxs + xPitch*tID);
		ys = (double*)((char*)hypoys + yPitch*tID);
		//QSortAlong(xs, ys, 'x', 0, width - 1);

		MergeSortAlongStream(xs, ys, 'x', tmpx, tmpy, width);
		xs[0] = inBd.xbeg;
		xs[width - 1] = inBd.xend;
		ys[0] = inBd.ybeg;
		ys[width - 1] = inBd.yend;

		tID += blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	}
	free(tmpx);
	tmpx = NULL;
	free(tmpy);
	tmpy = NULL;
}

__global__ void GetDataChordLenAndPStream(__in double* dataPtx, __in double* dataPty, __in double* dataPtz,__in double* dataH,
	__in size_t xPitch, __in size_t yPitch, __in size_t zPitch, __in size_t hPitch, __in unsigned int width, __in unsigned int height,
	__in size_t clPitch, __in size_t pPitch, __inout double* chordLen, __inout PointQuat* P)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;

	//Accumulated chord length parameterization for knot vector.
	unsigned int rowID = tID / width, colID = tID%width;
	double* xLine = (double*)((char*)dataPtx + rowID*xPitch),
		*yLine = (double*)((char*)dataPty + rowID*yPitch),
		*zLine = NULL,
		*HLine = (double*)((char*)dataH + rowID*hPitch),
		*clLine = (double*)((char*)chordLen + rowID*clPitch);
	PointQuat *PLine = (PointQuat*)((char*)P + rowID*pPitch);
	while (rowID<height && colID<width)
	{
		xLine = (double*)((char*)dataPtx + rowID*xPitch),
			yLine = (double*)((char*)dataPty + rowID*yPitch),
			//zLine = (double*)((char*)dataPtz + rowID*zPitch),
			HLine = (double*)((char*)dataH + rowID*hPitch),
			clLine = (double*)((char*)chordLen + rowID*clPitch);
		if (NULL != dataPtz)
		{
			zLine = (double*)((char*)dataPtz + rowID*hPitch);
		}
		else
		{
			zLine = NULL;
		}
		PLine = (PointQuat*)((char*)P + rowID*pPitch);

		if (colID < width - 1)
		{
			if (NULL != zLine)
			{
				clLine[colID] = sqrt((xLine[colID] - xLine[colID + 1])*(xLine[colID] - xLine[colID + 1]) +
					(yLine[colID] - yLine[colID + 1])*(yLine[colID] - yLine[colID + 1]) +
					(zLine[colID] - zLine[colID + 1])*(zLine[colID] - zLine[colID + 1]));
			}
			else
			{
				clLine[colID] = sqrt((xLine[colID] - xLine[colID + 1])*(xLine[colID] - xLine[colID + 1]) +
					(yLine[colID] - yLine[colID + 1])*(yLine[colID] - yLine[colID + 1]));
			}
		}
		HLine[colID] = 1.0;
		PLine[colID].x = xLine[colID] * HLine[colID];
		PLine[colID].y = yLine[colID] * HLine[colID];
		if (NULL != zLine)
		{
			PLine[colID].z = zLine[colID] * HLine[colID];
		}
		else
		{
			PLine[colID].z = 0.0;
		}
		
		PLine[colID].w = HLine[colID];

		tID += blockDim.x*blockDim.y*gridDim.x*gridDim.y;
		rowID = tID / width;
		colID = tID%width;
	}
}

__global__ void GetChordTotalLenStream(__inout double* chrodLen, __in size_t clPitch, __in unsigned int width, __in unsigned int height)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;

	double* chordLenLine = (double*)((char*)chrodLen + tID*clPitch);
	while (tID < height)
	{
		chordLenLine = (double*)((char*)chrodLen + tID*clPitch);
		for (unsigned int ii = 0; ii < width - 1; ii++)
		{
			chordLenLine[ii + 1] += chordLenLine[ii];
		}
		tID += blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	}
}

__global__ void GetUStream(__in double* chrodLen, __in size_t lenPitch, __in unsigned int width, __in unsigned int height, __in size_t UPitch, __inout double* U)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;
	
	unsigned int rowID = tID / (width + 6), colID = tID % (width + 6);
	double* tlLine = (double*)((char*)chrodLen + rowID*lenPitch),
		*ULine = (double*)((char*)U + rowID*UPitch);
	while (rowID < height && colID < (width + 6))
	{
		tlLine = (double*)((char*)chrodLen + rowID*lenPitch);
		ULine = (double*)((char*)U + rowID*UPitch);
		if (colID<4)
		{
			ULine[colID] = 0.0;
		}
		else if (colID > width + 1)
		{
			ULine[colID] = 1.0;
		}
		else
		{
			ULine[colID] = tlLine[colID - 4] / tlLine[width - 2];
		}

		tID += blockDim.x*blockDim.y*gridDim.x*gridDim.y;
		rowID = tID / (width + 6);
		colID = tID % (width + 6);
	}
}


__global__ void GetABCEStream(__in double* U, __in size_t UPitch, __in PointQuat* P, __in size_t PPitch,__in unsigned int width, __in unsigned int height,
	__in size_t APitch, __in size_t BPitch, __in size_t CPitch, __in size_t EPitch, __out double* A, __out double* B, __out double* C, __out PointQuat* E)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;

	unsigned int tmpID, rowID = tID , colID = 0;
	double* ALine = (double*)((char*)A + rowID*APitch),
		*BLine = (double*)((char*)B + rowID*BPitch),
		*CLine = (double*)((char*)C + rowID*CPitch),
		*ULine = (double*)((char*)U + rowID*UPitch);
	PointQuat* ELine = (PointQuat*)((char*)E + rowID*EPitch),
		*PLine = (PointQuat*)((char*)P + rowID*PPitch);

	//A,B,C,_C size 1*DataSize(Width), E,_E size 4*DataSize(Width)
	/* A,B,C are vectors which contain values of tri-diagonal matrix with the following form:
	|a1	b1	c1							|
	|a2	b2	c2							|
	|	a3	b3	c3						|
	|		...	...	...					|
	|			a_DS-1_	b_DS-1_	c_DS-1_	|
	|			a_DS_	b_DS_	c_DS_	|
	Therefore, A={a1,a2,...,a_DS_}, B={b1,b2,...,b_DS_} and C={c1,c2,...c_DS_}.
	In algorithm, the indices of vectors are start from 0, so A[i]=a_i+1_, for i=0,1,...DS-1
	*/

	while (rowID < height)//dataSize - 1 = n
	{
		ALine = (double*)((char*)A + rowID*APitch);
		BLine = (double*)((char*)B + rowID*BPitch);
		CLine = (double*)((char*)C + rowID*CPitch);
		ULine = (double*)((char*)U + rowID*UPitch);
		ELine = (PointQuat*)((char*)E + rowID*EPitch);
		PLine = (PointQuat*)((char*)P + rowID*PPitch);

		while (colID < width)
		{
			//Free end boundary condition
			if (colID == 0)
			{
				ALine[0] = 2 - (ULine[4] - ULine[3])*(ULine[5] - ULine[4]) / (ULine[5] - ULine[3]) / (ULine[5] - ULine[3]);
				BLine[0] = (ULine[4] - ULine[3]) / (ULine[5] - ULine[3])*((ULine[5] - ULine[4]) / (ULine[5] - ULine[3]) - (ULine[4] - ULine[3]) / (ULine[6] - ULine[3]));
				CLine[0] = (ULine[4] - ULine[3])*(ULine[4] - ULine[3]) / (ULine[5] - ULine[3]) / (ULine[6] - ULine[3]);
				ELine[0] = PLine[0] + PLine[1];
			}
			else if (colID == width - 1)
			{
				ALine[width - 1] = (ULine[width + 2] - ULine[width + 1])*(ULine[width + 2] - ULine[width + 1]) /
					(ULine[width + 2] - ULine[width]) / (ULine[width + 2] - ULine[width - 1]);
				BLine[width - 1] = (ULine[width + 2] - ULine[width + 1]) / (ULine[width + 2] - ULine[width]) *
					((ULine[width + 2] - ULine[width + 1]) / (ULine[width + 2] - ULine[width - 1]) - (ULine[width + 1] - ULine[width]) / (ULine[width + 2] - ULine[width]));
				CLine[width - 1] = (ULine[width + 1] - ULine[width])*(ULine[width + 2] - ULine[width + 1]) /
					(ULine[width + 2] - ULine[width]) / (ULine[width + 2] - ULine[width]) - 2;
				ELine[width - 1] = -1 * PLine[width] - PLine[width - 1];
			}//End of Free end boundary condition
			else
			{
				//i=1,2,...,dataSize,i=j+1,j=1,...,dataSize-2, dataSize=width
				//A[j]=(u_i+3-u_i+2)^2/(u_i+3-u_i)
				tmpID = colID + 1;
				ALine[colID] = (ULine[tmpID + 3] - ULine[tmpID + 2])*(ULine[tmpID + 3] - ULine[tmpID + 2]) / (ULine[tmpID + 3] - ULine[tmpID]);
				//B[j]=(u_i+3-u_i+2)*(u_i+2-u_i)/(u_i+3-u_i)+(u_i+2-u_i+1)*(u_i+4-u_i+2)/(u_i+4-u_i+1)
				BLine[colID] = (ULine[tmpID + 3] - ULine[tmpID + 2])*(ULine[tmpID + 2] - ULine[tmpID]) / (ULine[tmpID + 3] - ULine[tmpID]) +
					(ULine[tmpID + 2] - ULine[tmpID + 1])*(ULine[tmpID + 4] - ULine[tmpID + 2]) / (ULine[tmpID + 4] - ULine[tmpID + 1]);
				//C[j]=(u_i+2-u_i+1)^2/(u_i+4-u_i+1)
				CLine[colID] = (ULine[tmpID + 2] - ULine[tmpID + 1])*(ULine[tmpID + 2] - ULine[tmpID + 1]) / (ULine[tmpID + 4] - ULine[tmpID + 1]);
				//E[j]=(u_i+3-u_i+1)*P_i-1
				ELine[colID] = PLine[tmpID - 1] * (ULine[tmpID + 3] - ULine[tmpID + 1]);
			}

			colID++;
		}
		tID += blockDim.x*blockDim.y*gridDim.x*gridDim.y;
		rowID = tID;
	}
}

__global__ void GetControlPointsStream(__in double* dataPtx, __in double* dataPty, __in double* dataPtz, __in double* dataH,
	__in size_t xPitch, __in size_t yPitch, __in size_t zPitch, __in size_t hPitch, __in double* A, __in double* B, __in double* C, __in double* _C, __in PointQuat* E, __in PointQuat* _E,
	__in size_t APitch, __in size_t BPitch, __in size_t CPitch, __in size_t _CPitch, __in size_t EPitch, __in size_t _EPitch,
	__in unsigned int width, __in unsigned int height,
	__out PointQuat* ctrPts, __in size_t cpPitch)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;

	double __B0;
	PointQuat curD, pre1D, pre2D;
	unsigned int tmpID = 0, rowID = tID, colID = 0;
	double* xLine = (double*)((char*)dataPtx + rowID*xPitch),
		*yLine = (double*)((char*)dataPty + rowID*yPitch),
		*zLine = NULL,
		*hLine = (double*)((char*)dataH + rowID*hPitch);
	if (NULL != dataPtz)
	{
		zLine = (double*)((char*)dataPtz + rowID*zPitch);
	}
	double* ALine = (double*)((char*)A + rowID*APitch),
		*BLine = (double*)((char*)B + rowID*BPitch),
		*CLine = (double*)((char*)C + rowID*CPitch),
		*_CLine = (double*)((char*)_C + rowID*_CPitch);
	PointQuat* ELine = (PointQuat*)((char*)E + rowID*EPitch),
		*_ELine = (PointQuat*)((char*)_E + rowID*_EPitch),
		*ctrPtLine = (PointQuat*)((char*)ctrPts + rowID*cpPitch);

	while (rowID<height)
	{//Modified Tri-diagonal matrix
		xLine = (double*)((char*)dataPtx + rowID*xPitch);
		yLine = (double*)((char*)dataPty + rowID*yPitch);
		if (NULL != dataPtz)
		{
			zLine = (double*)((char*)dataPtz + rowID*zPitch);
		}
		hLine = (double*)((char*)dataH + rowID*hPitch);
		ALine = (double*)((char*)A + rowID*APitch);
		BLine = (double*)((char*)B + rowID*BPitch);
		CLine = (double*)((char*)C + rowID*CPitch);
		_CLine = (double*)((char*)_C + rowID*_CPitch);
		ELine = (PointQuat*)((char*)E + rowID*EPitch);
		_ELine = (PointQuat*)((char*)_E + rowID*_EPitch);
		ctrPtLine = (PointQuat*)((char*)ctrPts + rowID*cpPitch);

		__B0 = BLine[0] / ALine[0];
		_CLine[0] = CLine[0] / ALine[0];
		_CLine[1] = (CLine[1] * ALine[0] - ALine[1] * CLine[0]) / (BLine[1] * ALine[0] - ALine[1] * BLine[0]);
		_ELine[0] = ELine[0] / ALine[0];
		_ELine[1] = (ELine[1] - ALine[1] * ELine[0]) / (BLine[1] * ALine[0] - ALine[1] * BLine[0]);
		for (int ii = 2; ii < width - 1; ii++)
		{
			_CLine[ii] = CLine[ii] / (BLine[ii] - ALine[ii] * _CLine[ii - 1]);
			_ELine[ii] = (ELine[ii] - ALine[ii] * _ELine[ii - 1]) / (BLine[ii] - ALine[ii] * _CLine[ii - 1]);
		}
		tmpID = width - 1;
		_CLine[tmpID] = CLine[tmpID] / (BLine[tmpID] - ALine[tmpID] * _CLine[tmpID - 1]) - _CLine[tmpID];
		_ELine[tmpID] = (ELine[tmpID] - ALine[tmpID] * _ELine[tmpID - 2]) / (BLine[tmpID] - ALine[tmpID] * _CLine[tmpID - 2]) - _ELine[tmpID - 1];

		//Chasing method to solve linear system of tri-diagonal matrix
		ctrPtLine[width + 1].x = xLine[width - 1];
		ctrPtLine[width + 1].y = yLine[width - 1];
		if (NULL != zLine)
		{
			ctrPtLine[width + 1].z = zLine[width - 1];
		}
		else
		{
			ctrPtLine[width + 1].z = 0.0;
		}
		ctrPtLine[width + 1].w = hLine[width - 1];

		pre1D = _ELine[width - 1] / _CLine[width - 1];
		ctrPtLine[width] = pre1D.HomoCoordPtToPt3Dw();

		for (unsigned int ii = width - 2; ii >= 1; ii--)
		{
			curD = _ELine[ii] - pre1D*_CLine[ii];
			ctrPtLine[ii + 1] = curD.HomoCoordPtToPt3Dw();
			if (ii == 1)
			{
				pre2D = pre1D;
			}
			pre1D = curD;
		}
		curD = _ELine[1] - __B0 * pre1D - _CLine[0] * pre2D;
		ctrPtLine[1] = curD.HomoCoordPtToPt3Dw();

		ctrPtLine[0].x = xLine[0];
		ctrPtLine[0].y = yLine[0];
		if (NULL != zLine)
		{
			ctrPtLine[0].z = zLine[0];
		}
		else
		{
			ctrPtLine[0].z = 0.0;
		}
		ctrPtLine[0].w = hLine[0];

		tID += blockDim.x*blockDim.y*gridDim.x*gridDim.y;
		rowID = tID;
	}
}

__global__ void GetCurvePointAndCurvature3DStream(__in Point3Dw* ctrPts,__in size_t cpPitch, __in unsigned int ctrSize, __in size_t pcSize,__in size_t height, __in double* inU, __in size_t UPitch,
	__inout Point3D* cvPts,__in size_t cvPitch, __inout double* curvatures,__in size_t curPitch, __out Point3D* ptTans, __in size_t tanPitch, __out Point3D* ptNorms,__in size_t nmPitch)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;
	int p = 3,             //degree of basis function
		n = ctrSize - 1,  //(n+1) control points,id=0,1,...,n.
		m = n + p + 1;     //Size of knot vector(m+1),id=0,1,2,...,m.

	Point3D curCurve;
	int span = -1;// , prespan = -1;
	double* N = (double*)malloc((p + 1) * sizeof(double)),
		*left = (double*)malloc((p + 1) * sizeof(double)),
		*right = (double*)malloc((p + 1) * sizeof(double));

	double* Ndu = (double*)malloc((p + 1)*(p + 1) * sizeof(double)),
		*a = (double*)malloc(2 * (p + 1) * sizeof(double)),
		*ders2 = (double*)malloc(3 * (p + 1) * sizeof(double));
	double curCvt;
	Point3Dw ders[3], norms, tans, binorms;
	/*Point3Dw* ders = (Point3Dw*)malloc(3 * sizeof(Point3Dw)),
		*norms = (Point3Dw*)malloc(sizeof(Point3Dw)),
		*tans = (Point3Dw*)malloc(sizeof(Point3Dw)),
		*binorms = (Point3Dw*)malloc(sizeof(Point3Dw));*/

	unsigned int rowID = tID / pcSize, colID = tID%pcSize;
	double* curvLine = (double*)((char*)curvatures + rowID*curPitch);
	Point3D* cvLine = (Point3D*)((char*)cvPts + rowID*cvPitch),
		*tanLine = (Point3D*)((char*)ptTans + rowID*tanPitch),
		*normLine = (Point3D*)((char*)ptNorms + rowID*nmPitch);
	double u = (colID*1.0) / (pcSize*1.0);

	while (rowID<height && colID<pcSize)
	{
		curvLine = (double*)((char*)curvatures + rowID*curPitch);
		cvLine = (Point3D*)((char*)cvPts + rowID*cvPitch);
		tanLine = (Point3D*)((char*)ptTans + rowID*tanPitch);
		normLine = (Point3D*)((char*)ptNorms + rowID*nmPitch);

		FindSpan1(n, p, u, inU, &span);
		BasisFuns1(span, u, p, inU, left, right, N);
		CurvePoint3D(span, p, n, N, ctrPts, &curCurve);
		cvLine[colID] = curCurve;
		cvLine[colID].id = tID;

		CurvePointCurvature3D(p, m, inU, span, u, Ndu, ctrPts, left, right, a, ders2, ders, &curCvt, &norms, &tans, &binorms);

		curvLine[colID] = curCvt;

		if (NULL != tanLine)
		{
			tanLine[colID] = tans.HomoCoordPtToPt3D();
			tanLine[colID].id = tID;
		}

		if (NULL != normLine)
		{
			normLine[colID] = norms.HomoCoordPtToPt3D();
			normLine[colID].id = tID;
		}
		tID += blockDim.x*blockDim.y*gridDim.x*gridDim.y;
		rowID = tID / pcSize;
		colID = tID%pcSize;
		u = (colID*1.0) / (pcSize*1.0);
	}
	free(a);
	a = NULL;
	free(ders2);
	ders2 = NULL;
	free(Ndu);
	Ndu = NULL;
	free(N);
	N = NULL;
	free(left);
	left = NULL;
	free(right);
	right = NULL;
	/*free(ders);
	ders = NULL;
	free(norms);
	norms = NULL;
	free(tans);
	tans = NULL;
	free(binorms);
	binorms = NULL;*/
}


__global__ void ClearCurvePointsStream(__in GridProp ptProp, __in Point3D* cvPts,__in size_t cvPitch, __in size_t cvSize,__in size_t height, __out Point3D* outcvPts,__in size_t outcvPitch)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;

	unsigned int rowID = tID / cvSize, colID = tID%cvSize;
	double xLow = 0.0, xUpper = 0.0, yLow = 0.0, yUpper = 0.0;
	if (ptProp.xbeg < ptProp.xend)
	{
		xLow = ptProp.xbeg;
		xUpper = ptProp.xend;
	}
	else
	{
		xLow = ptProp.xend;
		xUpper = ptProp.xbeg;
	}

	if (ptProp.ybeg < ptProp.yend)
	{
		yLow = ptProp.ybeg;
		yUpper = ptProp.yend;
	}
	else
	{
		yLow = ptProp.yend;
		yUpper = ptProp.ybeg;
	}

	Point3D* outcvLine = (Point3D*)((char*)outcvPts + rowID*outcvPitch),
		*cvLine = (Point3D*)((char*)cvPts + rowID*cvPitch);
	while (rowID<height && colID<cvSize)
	{
		outcvLine = (Point3D*)((char*)outcvPts + rowID*outcvPitch);
		cvLine = (Point3D*)((char*)cvPts + rowID*cvPitch);

		if ((xLow - 1e-6) < cvLine[colID].x3 && cvLine[colID].x3 < (xUpper + 1e-6) &&
			(yLow - 1e-6) < cvLine[colID].y3 && cvLine[colID].y3 < (yUpper + 1e-6))
		{
			outcvLine[colID] = cvLine[colID];
		}
		else
		{
			outcvLine[colID].x3 = NAN;
			outcvLine[colID].y3 = NAN;
			outcvLine[colID].z3 = NAN;
		}

		tID += blockDim.x*blockDim.y*gridDim.x*gridDim.y;
		rowID = tID / cvSize;
		colID = tID%cvSize;
	}

}

__global__ void GetDataPointDistStream(__in double* hypox, __in size_t hypoxPitch, __in double* hypoy, __in size_t hypoyPitch,__in size_t hypoSize, __in double* dataPtx, __in double* dataPty, __in double* dataPtz, __in size_t xPitch, __in size_t yPitch, __in size_t zPitch, 
	__in Point3Dw* ctrPts, __in size_t cpPitch, __in unsigned int ctrSize, __in size_t pcSize, __in size_t height, __in double* inU, __in size_t UPitch,
	__in double* chordLen, __in size_t clPitch,__in size_t clSize, __out double * dists)
{
	unsigned int bID = blockIdx.x + blockIdx.y*gridDim.x,
		tID = threadIdx.x + threadIdx.y*blockDim.x + bID*blockDim.x*blockDim.y;
	int p = 3,             //degree of basis function
		n = ctrSize - 1,  //(n+1) control points,id=0,1,...,n.
		m = n + p + 1;     //Size of knot vector(m+1),id=0,1,2,...,m.

	Point3D curCurve;
	int span = -1;// , prespan = -1;
	double* N = (double*)malloc((p + 1) * sizeof(double)),
		*left = (double*)malloc((p + 1) * sizeof(double)),
		*right = (double*)malloc((p + 1) * sizeof(double));

	double* Ndu = (double*)malloc((p + 1)*(p + 1) * sizeof(double)),
		*a = (double*)malloc(2 * (p + 1) * sizeof(double)),
		*ders2 = (double*)malloc(3 * (p + 1) * sizeof(double));
	double curCvt;
	Point3Dw ders[3], norms, tans, binorms;

	unsigned int rowID = tID / pcSize, colID = tID%pcSize;

	double u = 0.0, ugap = 0.0, mingap = 1e10;
	double chordLenTotal = 0.0, uLen = 0.0;
	while (tID < pcSize)
	{
		double* hypoxLine = (double*)((char*)hypox + tID*hypoxPitch),
			*hypoyLine = (double*)((char*)hypoy + tID*hypoyPitch);
		for (unsigned int ii = 0; ii < hypoSize - 1; ii++)
		{
			if (hypoxLine[ii+1]<dataPtx[tID])
			{
				uLen += chordLen[ii];
			}
			else if (hypoxLine[ii] <= dataPtx[tID] && hypoxLine[ii + 1] >= dataPtx[tID])
			{
				uLen += chordLen[ii] * (dataPtx[tID] - hypoxLine[ii]) / (hypoxLine[ii + 1] - hypoxLine[ii]);
			}
			chordLenTotal += chordLen[ii];
		}
		u = uLen / chordLenTotal;
		do {
			FindSpan1(n, p, u, inU, &span);
			BasisFuns1(span, u, p, inU, left, right, N);
			CurvePoint3D(span, p, n, N, ctrPts, &curCurve);
			ugap = curCurve.x2 - dataPtx[tID];
			if (abs(ugap) < abs(mingap) && abs(ugap) > 1e-3)
			{
				mingap = ugap;
			}
			u += mingap / 2;
		} while (ugap > 1e-3);

		//CurvePointCurvature3D(p, m, inU, span, u, Ndu, ctrPts, left, right, a, ders2, ders, &curCvt, &norms, &tans, &binorms);

		tID += blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	}

	free(a);
	a = NULL;
	free(ders2);
	ders2 = NULL;
	free(Ndu);
	Ndu = NULL;
	free(N);
	N = NULL;
	free(left);
	left = NULL;
	free(right);
	right = NULL;
}


const char* NURBSRANSACOnGPUStream(__in CtrPtBound inBound, __in double* xvals, __in double* yvals, __in size_t pcSize, __in int maxIters, __in int minInliers, __in double UTh, __in double LTh,
	__out int*& resInliers, __out double &modelErr, __out double* &bestCtrx, __out double* &bestCtry, __out double* &bestDists)
{
	cudaError_t cudaErr;
	unsigned int gridX = 1, gridY = 1;
	dim3 blocks(32, 32), grids(gridX, gridY);
	cudaEvent_t begEvent, endEvent;
	cudaERRORHANDEL(cudaEventCreate(&begEvent), "begEvent Create");
	cudaERRORHANDEL(cudaEventCreate(&endEvent), "endEvent Create");
	float tEvent;

	//I.Copy total data from host into device.
	double* d_xs, *d_ys;
	cudaERRORHANDEL(cudaMalloc((void**)&d_xs, sizeof(double)*pcSize), "Malloc d_xs");
	cudaERRORHANDEL(cudaMalloc((void**)&d_ys, sizeof(double)*pcSize), "Malloc d_yx");
	cudaERRORHANDEL(cudaEventRecord(begEvent, 0), "begEvent Record");
	cudaERRORHANDEL(cudaMemcpy(d_xs, xvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice), "Copy d_xs H2D");
	cudaERRORHANDEL(cudaMemcpy(d_ys, yvals, sizeof(double)*pcSize, cudaMemcpyHostToDevice), "Copy d_ys H2D");
	cudaERRORHANDEL(cudaEventRecord(endEvent, 0), "endEvent Record");
	cudaERRORHANDEL(cudaEventSynchronize(endEvent), "endEvnet Sync");
	cudaERRORHANDEL(cudaEventElapsedTime(&tEvent, begEvent, endEvent), "Calculating Event Time");
	int curIt = 0;

	//II.Choose hypo-point x and ys randomly.
	//II.1 Generate uniform random values on GPU.
	float *d_Randf, *h_Randf;
	cudaERRORHANDEL(cudaMalloc((void**)&d_Randf, minInliers * maxIters * sizeof(float)), "Malloc d_Randf");
	cudaERRORHANDEL(cudaMallocHost((void**)&h_Randf, minInliers*maxIters * sizeof(float)), "Malloc h_Randf");
	curandGenerator_t gen;
	curandStatus_t curandErr;
	curandErr = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandErr = curandSetPseudoRandomGeneratorSeed(gen, unsigned long long(time(0)));
	curandErr = curandGenerateUniform(gen, d_Randf, minInliers * maxIters);

	//cudaErr = UserDebugCopy(h_Randf, d_Randf, minInliers * maxIters * sizeof(float), cudaMemcpyDeviceToHost);
	cudaERRORHANDEL(cudaFreeHost(h_Randf), "Free h_Randf");

	//II.2 Perpare variables to choose points randomly.
	size_t *d_HypoIDs, *h_HypoIDs;
	size_t hypoPitch;
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_HypoIDs, &hypoPitch, minInliers * sizeof(size_t), maxIters), "MallocPitch d_HypoIDs");
	cudaERRORHANDEL(cudaMemset2D(d_HypoIDs, hypoPitch, 0, minInliers * sizeof(size_t), maxIters), "Memset d_HypoIDs");
	cudaERRORHANDEL(cudaMallocHost((void**)&h_HypoIDs, minInliers*maxIters * sizeof(size_t)), "Malloc h_HypoIDs");

	double *d_HypoXs, *d_HypoYs, *h_HypoXs, *h_HypoYs;
	size_t hypoXPitch, hypoYPitch;
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_HypoXs, &hypoXPitch, minInliers * sizeof(double), maxIters), "Malloc d_HypoXs");
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_HypoYs, &hypoYPitch, minInliers * sizeof(double), maxIters), "Malloc d_HypoYs");
	cudaERRORHANDEL(cudaMallocHost((void**)&h_HypoXs, minInliers*maxIters * sizeof(double)), "Malloc h_HypoXs");
	cudaERRORHANDEL(cudaMallocHost((void**)&h_HypoYs, minInliers*maxIters * sizeof(double)), "Malloc h_HypoYs");

	//II.3 Setup streams for choosing kernels on GPU.
	unsigned int nStream = 8;
	cudaStream_t* Streams = (cudaStream_t*)malloc(nStream * sizeof(cudaStream_t));
	for (unsigned int ii = 0; ii < nStream; ii++)
	{
		cudaERRORHANDEL(cudaStreamCreate(&Streams[ii]), "Create Stream");
	}
	blocks = dim3(32, 32);
	cudaERRORHANDEL(cudaEventRecord(begEvent, 0), "begEvent Record d_HypoIDs");
	unsigned int offsetRowsBase = (maxIters + nStream - 1) / nStream, useRows = offsetRowsBase, offsetRows;
	unsigned int useGrid = 10 / nStream;

	//II.4 Start selecting points randomly on GPU by streams.
	for (unsigned int ii = 0; ii < nStream; ii++)
	{
		offsetRows = ii*offsetRowsBase;
		if (ii == nStream - 1)
		{
			useRows = maxIters - ii*offsetRowsBase;
			if (10 % nStream > 0)
			{
				useGrid = 10 % nStream;
			}
		}

		ConvertF2IMatStream << <useGrid, blocks, 0, Streams[ii] >> > (&d_Randf[offsetRows*minInliers], useRows*minInliers, hypoPitch, minInliers, pcSize, (size_t*)((char*)d_HypoIDs + offsetRows*hypoPitch));
		cudaERRORHANDEL(cudaGetLastError(), "ConvertF2IMatStream Error");
		//cudaERRORHANDEL(cudaMemcpy2DAsync(&h_HypoIDs[offsetRows*minInliers], minInliers * sizeof(size_t), (size_t*)((char*)d_HypoIDs + offsetRows*hypoPitch), hypoPitch, minInliers * sizeof(size_t), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Stream D2H Copy h_HypoIDs");
		GetHypoXYAtOnceStream << <useGrid, blocks, 0, Streams[ii] >> > (d_xs, d_ys,(size_t*)((char*)d_HypoIDs+offsetRows*hypoPitch), hypoPitch, minInliers, useRows, (double*)((char*)d_HypoXs+offsetRows*hypoXPitch), hypoXPitch, (double*)((char*)d_HypoYs+offsetRows*hypoYPitch), hypoYPitch);
		cudaERRORHANDEL(cudaGetLastError(), "GetHypoXYAtOnceStream Error");
		//cudaERRORHANDEL(cudaMemcpy2DAsync(&h_HypoXs[offsetRows*minInliers], minInliers * sizeof(double), (double*)((char*)d_HypoXs + offsetRows*hypoXPitch), hypoXPitch, minInliers * sizeof(double), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Stream D2H Copy h_HypoXs");
		//cudaERRORHANDEL(cudaMemcpy2DAsync(&h_HypoYs[offsetRows*minInliers], minInliers * sizeof(double), (double*)((char*)d_HypoYs + offsetRows*hypoYPitch), hypoYPitch, minInliers * sizeof(double), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Stream D2H Copy h_HypoYs");
		SortHypoXYStream <<<useGrid, blocks, 0, Streams[ii] >>> (inBound, minInliers, useRows, (double*)((char*)d_HypoXs + offsetRows*hypoXPitch), hypoXPitch, (double*)((char*)d_HypoYs + offsetRows*hypoYPitch), hypoYPitch);
		cudaERRORHANDEL(cudaGetLastError(), "SortHypoXYStream Error");
		//cudaERRORHANDEL(cudaGetLastError(), "Run SortHypoXYStream");
		//cudaERRORHANDEL(cudaMemcpy2DAsync(&h_HypoXs[offsetRows*minInliers], minInliers * sizeof(double), (double*)((char*)d_HypoXs + offsetRows*hypoXPitch), hypoXPitch, minInliers * sizeof(double), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Stream D2H Copy h_HypoXs");
		//cudaERRORHANDEL(cudaMemcpy2DAsync(&h_HypoYs[offsetRows*minInliers], minInliers * sizeof(double), (double*)((char*)d_HypoYs + offsetRows*hypoYPitch), hypoYPitch, minInliers * sizeof(double), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Stream D2H Copy h_HypoYs");
	}

	//II.5 End of GPU selecting points method and free some host pin-paged spaces.
	cudaERRORHANDEL(cudaDeviceSynchronize(), "Stream Sync d_HypoIDs");
	cudaERRORHANDEL(cudaEventRecord(endEvent, 0), "endEvent Record d_HypoIDs");
	cudaERRORHANDEL(cudaEventSynchronize(endEvent), "endEvnet Sync d_HypoIDs");
	cudaERRORHANDEL(cudaEventElapsedTime(&tEvent, begEvent, endEvent), "Calculating Event Time d_HypoIDs");

	cudaERRORHANDEL(cudaFree(d_Randf), "Free d_Randf");
	cudaERRORHANDEL(cudaFreeHost(h_HypoIDs), "Free h_HypoIDs");
	cudaERRORHANDEL(cudaFreeHost(h_HypoXs), "Free h_HypoXs");
	cudaERRORHANDEL(cudaFreeHost(h_HypoYs), "Free h_HypoYs");

	//III. Generating control points from hypo data points.
	//III.1 Setup some variables for GPU kernels.
	PointQuat* d_P, *h_P;
	size_t pPitch;
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_P, &pPitch, minInliers * sizeof(PointQuat), maxIters), "d_P Malloc");
	cudaERRORHANDEL(cudaMallocHost((void**)&h_P, minInliers*maxIters * sizeof(PointQuat)), "h_P Malloc");
	double* d_ChrodLen, *h_ChrodLen, *d_U, *h_U, *d_H;
	size_t clPitch, UPitch, hPitch;
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_H, &hPitch, minInliers * sizeof(double), maxIters), "MallocPitch H");
	cudaERRORHANDEL(cudaMallocHost((void**)&h_U, (minInliers + 6)*maxIters * sizeof(double)), "Malloc h_U");
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_U, &UPitch, (minInliers + 6) * sizeof(double), maxIters), "d_U Malloc");
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_ChrodLen, &clPitch, (minInliers - 1) * sizeof(double), maxIters), "d_ChrodLen Malloc");
	cudaERRORHANDEL(cudaMallocHost((void**)&h_ChrodLen, (minInliers - 1)*maxIters * sizeof(double)), "cudaMalloc h_ChrodLen");

	double* d_A, *d_B, *d_C, *d__C;
	size_t APitch, BPitch, CPitch, _CPitch;
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_A, &APitch, minInliers * sizeof(double), maxIters), "d_A Malloc");
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_B, &BPitch, minInliers * sizeof(double), maxIters), "d_B Malloc");
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_C, &CPitch, minInliers * sizeof(double), maxIters), "d_C Malloc");
	cudaERRORHANDEL(cudaMallocPitch((void**)&d__C, &_CPitch, minInliers * sizeof(double), maxIters), "d__C Malloc");

	PointQuat* d_E, *d__E, *d_ctrPts, *h_ctrPts;
	size_t EPitch, _EPitch, cpPitch;
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_E, &EPitch, minInliers * sizeof(PointQuat), maxIters), "d_E Malloc");
	cudaERRORHANDEL(cudaMallocPitch((void**)&d__E, &_EPitch, minInliers * sizeof(PointQuat), maxIters), "d__E Malloc");
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_ctrPts, &cpPitch, (minInliers + 2) * sizeof(PointQuat), maxIters), "d_ctrPts Malloc");
	cudaERRORHANDEL(cudaMallocHost((void**)&h_ctrPts, (minInliers + 2)*maxIters * sizeof(PointQuat)), "h_ctrPts Malloc");

	cudaERRORHANDEL(cudaEventRecord(begEvent, 0), "begEvent Record control point generating");

	useGrid = 10 / nStream;
	//III.2 Run through streams to generate contral ponits.
	for (unsigned int ii = 0; ii < nStream; ii++)
	{
		offsetRows = ii*offsetRowsBase;
		if (ii == nStream - 1)
		{
			useRows = maxIters - ii*offsetRowsBase;
			if (10 % nStream > 0)
			{
				useGrid = 10 % nStream;
			}
		}

		GetDataChordLenAndPStream << <useGrid, 512, 0, Streams[ii] >> > ((double*)((char*)d_HypoXs + offsetRows*hypoXPitch), (double*)((char*)d_HypoYs + offsetRows*hypoYPitch), NULL, d_H, hypoXPitch, hypoYPitch, 0, hPitch,
			minInliers, useRows, clPitch, pPitch, (double*)((char*)d_ChrodLen + offsetRows*clPitch), (PointQuat*)((char*)d_P + offsetRows*pPitch));
		cudaERRORHANDEL(cudaGetLastError(), "GetDataChordLenAndPStream Error");
		//cudaERRORHANDEL(cudaMemcpy2DAsync(&h_ChrodLen[offsetRows*(minInliers - 1)], (minInliers - 1) * sizeof(double), (double*)((char*)d_ChrodLen + offsetRows*clPitch), clPitch, (minInliers - 1) * sizeof(double), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Stream D2H Copy h_ChrodLen");
		//cudaERRORHANDEL(cudaMemcpy2DAsync(&h_P[offsetRows*minInliers], minInliers * sizeof(PointQuat), (PointQuat*)((char*)d_P + offsetRows*clPitch), pPitch, minInliers* sizeof(PointQuat), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Stream D2H Copy h_P");

		GetChordTotalLenStream << <useGrid, blocks, 0, Streams[ii] >> > ((double*)((char*)d_ChrodLen + offsetRows*clPitch), clPitch, (minInliers - 1), useRows);
		cudaERRORHANDEL(cudaGetLastError(), "GetChordTotalLenStream Error");
		//cudaERRORHANDEL(cudaMemcpy2DAsync(&h_ChrodLen[offsetRows*(minInliers - 1)], (minInliers - 1) * sizeof(double), (double*)((char*)d_ChrodLen + offsetRows*clPitch), clPitch, (minInliers - 1) * sizeof(double), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Stream D2H Copy h_ChrodLen");

		GetUStream << <useGrid, blocks, 0, Streams[ii] >> >((double*)((char*)d_ChrodLen + offsetRows*clPitch), clPitch, minInliers, useRows, UPitch, (double*)((char*)d_U + offsetRows*UPitch));
		cudaERRORHANDEL(cudaGetLastError(), "GetUStream Error");
		//cudaERRORHANDEL(cudaMemcpy2DAsync(&h_U[offsetRows*(minInliers+6)], (minInliers+6) * sizeof(double), (double*)((char*)d_U + offsetRows*UPitch), UPitch, (minInliers+6) * sizeof(double), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Stream D2H Copy h_U");

		GetABCEStream << <useGrid, 512, 0, Streams[ii] >> > ((double*)((char*)d_U + offsetRows*UPitch), UPitch, (PointQuat*)((char*)d_P + offsetRows*pPitch), pPitch, minInliers, useRows,
			APitch, BPitch, CPitch, EPitch, (double*)((char*)d_A + offsetRows*APitch), (double*)((char*)d_B + offsetRows*BPitch), (double*)((char*)d_C + offsetRows*CPitch), (PointQuat*)((char*)d_E + offsetRows*EPitch));
		cudaErr = cudaGetLastError();
		cudaERRORHANDEL(cudaErr, "GetABCEStream");

		GetControlPointsStream << <useGrid, 256, 0, Streams[ii] >> > ((double*)((char*)d_HypoXs + offsetRows*hypoXPitch), (double*)((char*)d_HypoYs + offsetRows*hypoYPitch), NULL, (double*)((char*)d_H + offsetRows*hPitch), hypoXPitch, hypoYPitch, 0, hPitch,
			(double*)((char*)d_A + offsetRows*APitch), (double*)((char*)d_B + offsetRows*BPitch), (double*)((char*)d_C + offsetRows*CPitch), (double*)((char*)d__C + offsetRows*_CPitch), (PointQuat*)((char*)d_E + offsetRows*EPitch), (PointQuat*)((char*)d__E + offsetRows*_EPitch),
			APitch, BPitch, CPitch, _CPitch, EPitch, _EPitch, minInliers, useRows, (PointQuat*)((char*)d_ctrPts + offsetRows*cpPitch), cpPitch);
		cudaErr = cudaGetLastError();
		cudaERRORHANDEL(cudaErr, "GetControlPointsStream");
		cudaERRORHANDEL(cudaMemcpy2DAsync(&h_ctrPts[offsetRows*(minInliers + 2)], (minInliers + 2) * sizeof(PointQuat), (PointQuat*)((char*)d_ctrPts + offsetRows*cpPitch), cpPitch, (minInliers + 2) * sizeof(PointQuat), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Stream D2H Copy h_ctrPts");
	}

	//III.3 End of GPU kernels for control points generation on streamings, and free some host and device spaces as well.
	cudaERRORHANDEL(cudaDeviceSynchronize(), "Stream Sync control point generating");
	cudaERRORHANDEL(cudaEventRecord(endEvent, 0), "endEvent Record control point generating");
	cudaERRORHANDEL(cudaEventSynchronize(endEvent), "endEvnet Sync control point generating");
	cudaERRORHANDEL(cudaEventElapsedTime(&tEvent, begEvent, endEvent), "Calculating Event Time control point generating");

	cudaERRORHANDEL(cudaFreeHost(h_ChrodLen), "Free h_ChrodLen");
	cudaERRORHANDEL(cudaFreeHost(h_P), "Free h_P");
	cudaERRORHANDEL(cudaFreeHost(h_U), "Free h_U");
	cudaERRORHANDEL(cudaFree(d_HypoIDs), "Free d_HypoIDs");
	cudaERRORHANDEL(cudaFree(d_HypoXs), "Free d_HypoXs");
	cudaERRORHANDEL(cudaFree(d_HypoYs), "Free d_HypoYs");
	cudaERRORHANDEL(cudaFreeHost(h_ctrPts), "Free h_ctrPts");

	cudaERRORHANDEL(cudaFree(d_A), "Free d_A");
	cudaERRORHANDEL(cudaFree(d_B), "Free d_B");
	cudaERRORHANDEL(cudaFree(d_C), "Free d_C");
	cudaERRORHANDEL(cudaFree(d__C), "Free d__C");
	cudaERRORHANDEL(cudaFree(d_E), "Free d_E");
	cudaERRORHANDEL(cudaFree(d__E), "Free d__E");

	//IV. Generating NURBRS curves and checking models.
	//IV.1 Perparing such parameters will be usied in kernel.
	double* d_curvatures;
	size_t curvPitch, cvSize = 5000;
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_curvatures, &curvPitch, cvSize * sizeof(double), maxIters), "Malloc d_curvatures");

	Point3D* d_cvPoint,*h_cvPoint, *d_ptTans, *d_ptNorms, *d_valcvPt, *h_valcvPt;
	size_t cvPitch, tanPitch, normPitch, valPitch;
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_cvPoint, &cvPitch, cvSize * sizeof(Point3D), maxIters), "Malloc d_cvPoint");
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_ptTans, &tanPitch, cvSize * sizeof(Point3D), maxIters), "Malloc d_ptTans");
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_ptNorms, &normPitch, cvSize * sizeof(Point3D), maxIters), "Malloc d_ptNorms");
	cudaERRORHANDEL(cudaMallocPitch((void**)&d_valcvPt, &valPitch, cvSize * sizeof(Point3D), maxIters), "Malloc d_valcvPt");
	cudaERRORHANDEL(cudaMallocHost((void**)&h_cvPoint, cvSize*maxIters * sizeof(Point3D)), "Malloc h_cvPoint");
	cudaERRORHANDEL(cudaMallocHost((void**)&h_valcvPt, cvSize*maxIters * sizeof(Point3D)), "Malloc h_valcvPt");

	cudaERRORHANDEL(cudaEventRecord(begEvent, 0), "begEvent Record curve point generating");
	for (unsigned int ii = 0; ii < nStream; ii++)
	{
		cudaERRORHANDEL(cudaStreamDestroy(Streams[ii]), "Destroy Stream");
	}
	free(Streams);
	nStream = 4;
	Streams = (cudaStream_t*)malloc(nStream * sizeof(cudaStream_t));
	for (unsigned int ii = 0; ii < nStream; ii++)
	{
		cudaERRORHANDEL(cudaStreamCreate(&Streams[ii]), "Create Stream");
	}
	offsetRowsBase = (maxIters + nStream - 1) / nStream;
	useRows = offsetRowsBase;
	blocks = dim3(32, 16);
	unsigned int totalGrid = 10240 / blocks.x / blocks.y;
	useGrid = totalGrid / nStream;
	//IV.2 Run through streams to generate curve ponits.
	for (unsigned int ii = 0; ii < nStream; ii++)
	{
		offsetRows = ii*offsetRowsBase;
		if (ii == nStream - 1)
		{
			useRows = maxIters - ii*offsetRowsBase;
			if (totalGrid % nStream > 0)
			{
				useGrid = totalGrid % nStream;
			}
		}
		GetCurvePointAndCurvature3DStream <<<useGrid, blocks, 0, Streams[ii] >>> ((Point3Dw*)((char*)d_ctrPts + offsetRows*cpPitch), cpPitch, minInliers + 2, cvSize, useRows, (double*)((char*)d_U + offsetRows*UPitch), UPitch,
			(Point3D*)((char*)d_cvPoint + offsetRows*cvPitch), cvPitch, (double*)((char*)d_curvatures + offsetRows*curvPitch), curvPitch, (Point3D*)((char*)d_ptTans + offsetRows*tanPitch), tanPitch,
			(Point3D*)((char*)d_ptNorms + offsetRows*normPitch), normPitch);
		cudaERRORHANDEL(cudaGetLastError(), "GetCurvePointAndCurvature3DStream");
		cudaERRORHANDEL(cudaMemcpy2DAsync(&h_cvPoint[offsetRows*cvSize], cvSize * sizeof(Point3D), (Point3D*)((char*)d_cvPoint + offsetRows*cvPitch), cvPitch, cvSize*sizeof(Point3D), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Copy h_cvPoint");


		ClearCurvePointsStream <<<useGrid, blocks, 0, Streams[ii] >>> (inBound, (Point3D*)((char*)d_cvPoint + offsetRows*cvPitch), cvPitch, cvSize, useRows, (Point3D*)((char*)d_valcvPt + offsetRows*valPitch), valPitch);
		cudaERRORHANDEL(cudaGetLastError(), "ClearCurvePointsStream");
		cudaERRORHANDEL(cudaMemcpy2DAsync(&h_valcvPt[offsetRows*cvSize], cvSize * sizeof(Point3D), (Point3D*)((char*)d_valcvPt + offsetRows*valPitch), valPitch, cvSize*sizeof(Point3D), useRows, cudaMemcpyDeviceToHost, Streams[ii]), "Copy h_valcvPt");
	}

	//IV.3 End of GPU kernels for curve points generation on streamings, and free some host and device spaces as well.
	cudaERRORHANDEL(cudaDeviceSynchronize(), "Stream Sync curve point generating");
	cudaERRORHANDEL(cudaEventRecord(endEvent, 0), "endEvent Record curve point generating");
	cudaERRORHANDEL(cudaEventSynchronize(endEvent), "endEvnet Sync curve point generating");
	cudaERRORHANDEL(cudaEventElapsedTime(&tEvent, begEvent, endEvent), "Calculating Event Time curve point generating");
	//V. Free all malloc spaces
	cudaERRORHANDEL(cudaFree(d_xs), "Free d_x");
	cudaERRORHANDEL(cudaFree(d_ys), "Free d_y");

	cudaERRORHANDEL(cudaFree(d_ChrodLen), "Free d_ChrodLen");
	cudaERRORHANDEL(cudaFree(d_P), "Free d_P");
	cudaERRORHANDEL(cudaFree(d_H), "Free d_H");

	for (unsigned int ii = 0; ii < nStream; ii++)
	{
		cudaERRORHANDEL(cudaStreamDestroy(Streams[ii]), "Free Stream");
	}

	free(Streams);
	Streams = NULL;
	

	return "Success";
}
