//2018/12/3 TonyHE Create file 
//2018/12/3 TonyHE Add enum TPCStatus and class TyrePointCloud
//2018/12/9 TonyHE Add Segementation searching method
//2018/12/10 TonyHE Add Properties for SegSearching
//2018/12/12 TonyHE Add FiltePins overload function
//2018/12/13 TonyHE Add FindPins function with input of char pointer.
/*	2019/2/28 TonyHE Verstion 1.0 of TPC with CUDA
	1.Using RANSAC algorithm to split tyre point cloud into two parts: 
	the segement basement part and the characterist part.
	2.Also depending on PCL to load and save point cloud, this method 
	will be deprecated in further versions.
*/
#pragma once
#undef min
#undef max

#define ACCURACY 10e-6

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

#include "cudaMain.h"

#include <pcl\io\pcd_io.h>
#include <pcl\io\ply_io.h>
#include <pcl\io\io.h>

#include <pcl\common\common.h>

#include <pcl\point_types.h>
#include <pcl\point_cloud.h>
#include <pcl\features\normal_3d.h>
#include <pcl\features\normal_3d_omp.h>
#include <pcl\common\transforms.h>
#include <pcl\filters\extract_indices.h>
#include <pcl\filters\voxel_grid.h>
#include <pcl\sample_consensus\method_types.h>
#include <pcl\segmentation\sac_segmentation.h>
#include <pcl\segmentation\extract_clusters.h>
#include <pcl\ModelCoefficients.h>

#include <pcl\segmentation\supervoxel_clustering.h>
#include <pcl\segmentation\lccp_segmentation.h>
#include <pcl\segmentation\cpc_segmentation.h>

#include <pcl\io\png_io.h>
#include <pcl\io\io.h>
#include <pcl\common\io.h>

#include <pcl\gpu\features\features.hpp>
#include <pcl\gpu\octree\device_format.hpp>
#include <pcl\gpu\containers\device_memory.hpp>
#include <pcl\gpu\containers\device_array.hpp>
#include <pcl\gpu\octree\octree.hpp>
//#include <pcl\gpu\segmentation\impl\gpu_extract_clusters.hpp>
//#include <pcl\gpu\segmentation\gpu_extract_clusters.h>
#include <boost\shared_ptr.hpp>

#include <ppl.h>

#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <thrust\functional.h>

#include "cuda_runtime.h"


using namespace pcl;
using namespace Eigen;
using namespace std;
using namespace concurrency;

enum TPCStatus
{
	//File and Point cloud loading
	EMPTY_FILE_NAME = -1,
	LOAD_PLY_ERROR = -2,
	LOAD_PCD_ERROR = -3,
	FILE_TYPE_ERROR = -4,
	NULL_PC_PTR = -5,
	EMPTY_POINT = -6,
	EMPTY_CHAR_PTR = -7,
	LOAD_CHAR_ERROR = -8,

	//Estimating normals
	NEGATIVE_R_K = -50,

	//Segementation 
	NONE_INLIERS = -100
};

struct PinObject
{
	float x;
	float y;
	float z;
	float len;
};

enum PtRefType//Point reference type
{
	//_P is for pointer and _V for vector.
	ORIGIN_P,
	DOWNSAMPLE_P,
	SEGEMENTBASE_P,
	REFPLANES_V,
	REFCOEFFICIENTS_V,
	RESTCLUSTERS_V,
	RESTINDICES_V,
	CANDIDATEPINS_P,
	PINSPC_P,
	PINSID_V,
	ORIGINRGB_P,
	POINTNORMAL_P
};



//Class for point cloud
class TyrePointCloud
{
public:
	TyrePointCloud();
	~TyrePointCloud();

private:
	BOOL m_normalUserDefined;

	//GPU Point Normal Searching Properties:
	float m_searchradius;
	int m_maxneighboranswers;
	
	//Segementation Properties:
	unsigned int m_numberofthreds;
	double m_distancethreshold;
	double m_normaldistanceweight;
	double m_inlierratio;
	float m_clustertolerance;
	double m_segmaxradius;
	double m_segminradius;

	PinObject m_pinsobj;

	//Implementation
	

protected:
	char* m_inCloud;//The input point cloud char pointer.

	PointCloud<PointXYZ>::Ptr m_originPC;//Original point cloud
	PointCloud<PointXYZ>::Ptr m_downsample;//Original down sampling cloud.
	PointCloud<PointXYZ>::Ptr m_segbase;//Segementation basic cloud.
	PointCloud<PointXYZ>::Ptr m_restPC;//Rest point cloud after segementation.
	vector<PointCloud<PointXYZ>::Ptr> m_refPlanes;//Reference planes' list.
	vector<ModelCoefficients::Ptr> m_refCoefs;//The coefficients of planes.
	vector<PointCloud<PointXYZI>::Ptr> m_restClusters;//The clusters after segmentation searching
	vector<PointIndices> m_restIndices;// The indices of clusters.
	PointCloud<PointXYZI>::Ptr m_candPins;//Candidate pins' point cloud.
	PointCloud<PointXYZI>::Ptr m_pinsPC;//Pins on the tyres, include positions(X,Y,Z) and length(I).
	vector<int> m_pinsID;// The head of pins' id in origin point cloud.
	PointCloud<PointXYZRGB>::Ptr m_rgbPC;//Point cloud with RGB.
	PointCloud<Normal>::Ptr m_pointNormals;//Point nomrals.
	PointCloud<Normal>::Ptr m_gpuPtNormals;//Point normals by applying GPU computing.

	map<int, vector<int>> m_clusterDic;//Key is the ID in refPlanes/refCoefs vector
									   //value is the ID vector of restClusters

	int FiltPins(Vector3d mineigenVector, vector<PointXYZI>& filted_pins);
	int FiltPins(vector<PointXYZI>& filted_pins);
	int SupervoxelClustering(pcl::PointCloud<PointXYZ>::Ptr in_pc,
		std::map <uint32_t, pcl::Supervoxel<PointXYZ>::Ptr > &sv_cluster,
		std::multimap<uint32_t, uint32_t>& sv_adj,
		pcl::PointCloud<PointXYZL>::Ptr & lbl_pc,
		float voxel_res = 0.03f, float seed_res = 0.09f, float color_imp = 0.0f,
		float spatial_imp=0.6f, float normal_imp=1.0f);

public:
	void InitCloudData();
	//Get and Set point clouds.
	void SetOriginPC(PointCloud<PointXYZ>::Ptr in_pc);
	void setOriginRGBPC(PointCloud<PointXYZRGB>::Ptr in_pc);

	PointCloud<PointXYZ>::Ptr GetOriginalPC();
	PointCloud<PointXYZ>::Ptr GetSegPC();
	PointCloud<PointXYZ>::Ptr GetRestPC();
	PointCloud<PointXYZI>::Ptr GetPinsPC();
	PointCloud<PointXYZRGB>::Ptr GetRGBPC();
	PointCloud<Normal>::Ptr GetPointNormals();
	PointCloud<Normal>::Ptr GetGPUPointNormals();

	//Get segementation references and clusters.
	void GetReferencePlanes(vector<PointCloud<PointXYZ>::Ptr> &out_ref);
	void GetReferenceCoefficients(vector<ModelCoefficients::Ptr> &out_ref);
	void GetRestClusters(vector<PointCloud<PointXYZI>::Ptr> &out_ref);

public:
	//Set parameters:
	//Point Normal Searching
	void SetSearchRadius(float radius);
	void SetMaxNeighborAnswers(int maxans);

	//Segmentation
	void SetNumberOfThreads(unsigned int nt);
	void SetDistanceThreshold(double dt);
	void SetNormalDistanceWeight(double ndw);
	void SetInlierRatio(double ir);
	void SetClusterTolerance(double ct);

public:
	int LoadTyrePC(string pcfile, float xLB = 0.0f, float xUB = 1000.0f, float yStep=0.03f, float zLB=0.0f, float zUB=1000.0f, size_t width = 1536, size_t height = 10000,float xBeg=0, float xEnd=1000);
	/* Loading tyre point clouds from file, which contains in .ply, .pcd or .dat file.
	   Parameters:
	     pcfile(in): The input file directory of point clouds.
		 NOTE: the following three parameters are only avalible for .dat file, the point size is width*height.
		 xLB(in): The lower bound along x-axis.
		 xUP(in): The upper bound along x-axis.
		 yStep(in): The step length along y-axis.
		 zLB(in): The lower bound along z-axis.
		 zUP(in): The upper bound along z-axis.
		 width(in): The width of a laser line in Range Image.
		 height(in): The height of Range Image.
	*/

	int FindPointNormalsGPU(PointCloud<PointXYZ>::Ptr in_pc, pcl::gpu::Octree::Ptr &in_tree, PointCloud<Normal>::Ptr &out_normal);

	int FindPinsBySegmentationGPU(PointCloud<PointXYZ>::Ptr in_pc, PointCloud<PointXYZI>::Ptr &out_pc);

	int FindCharsBySegmentationGPU(PointCloud<PointXYZ>::Ptr in_pc, PointCloud<PointXYZ>::Ptr &out_pc);

	int FindCharsByLCCP(pcl::PointCloud<PointXYZ>::Ptr in_pc, pcl::PointCloud<PointXYZL>::Ptr &out_pc);
	/* LCCP: Locally Convex Connected Patches
	   This method has two steps: 1. Supervoxel Clustering, 2. Convex Connected Clustering.
	*/

	int FindCharsByCPC(pcl::PointCloud<PointXYZ>::Ptr in_pc, pcl::PointCloud<PointXYZL>::Ptr &out_pc);

	int FindPins(char* p_pc, int length, vector<PinObject> & out_pc);
	/* Find pins by input a char stream
	   Parameters:
	     p_pc(in): Char pointer of the input point cloud.
		 length(in): The length of char stream.
		 out_pc(out): A list of PinObjects, which have position(x,y,z) and the length(len).
	*/

	int FindCharsBy2DRANSACGPU(pcl::PointCloud<PointXYZ>::Ptr in_pc, int maxIters, int minInliers, int paraSize, double UTh, double LTh,
		pcl::PointCloud<PointXYZ>::Ptr & char_pc, pcl::PointCloud<PointXYZ>::Ptr & base_pc);
	/* This function splits one in put point cloud(in_pc) into two parts: PC including chars(char_pc) and the basement
	   one(base_pc).
	   Parameters:
		in_pt(in): Input point cloud pointer.
		maxIters(in): The iteration numbers for RANSAC.
		minInliers(in): The number of points(called inliers) for the beginning of iteration.
		paraSize(in): The parameters of fitted polynomials in 2D, where paraSize=order+1.
		UTh(in): The upper boundary of threshold for fitted polynomials.
		LTh(in): The lower boundary of threshold for fitted polynomials.
		char_pc(out): The pointer of point cloud contains characteristics.
		base_pc(out): The pointer of point cloud includes the basement points.
	*/

};