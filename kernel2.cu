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