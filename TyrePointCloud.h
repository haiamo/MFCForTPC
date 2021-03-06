//2018/12/3 TonyHE Create file 
//2018/12/3 TonyHE Add enum TPCStatus and class TyrePointCloud
//2018/12/9 TonyHE Add Segementation searching method
//2018/12/10 TonyHE Add Properties for SegSearching
//2018/12/12 TonyHE Add FiltePins overload function
//2018/12/13 TonyHE Add FindPins function with input of char pointer.
#pragma once
#undef min
#undef max

#define ACCURACY 10e-6

#include <iostream>
 
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

#include <ppl.h>

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

class TyrePointCloud
{
public:
	TyrePointCloud();
	~TyrePointCloud();

private:
	BOOL m_normalUserDefined;

	//Segementation Properties:
	float m_downsampleradius;
	unsigned int m_numberofthreds;
	double m_distancethreshold;
	double m_normaldistanceweight;
	double m_inlierratio;
	float m_clustertolerance;

	PinObject m_pinsobj;

	//Implementation
	

protected:
	char* m_inCloud;//The input point cloud char pointer.

	PointCloud<PointXYZ>::Ptr m_originPC;//Original point cloud
	PointCloud<PointXYZ>::Ptr m_downsample;//Original down sampling cloud.
	PointCloud<PointXYZ>::Ptr m_segbase;//Segementation basic cloud.
	vector<PointCloud<PointXYZ>::Ptr> m_refPlanes;//Reference planes' list.
	vector<ModelCoefficients::Ptr> m_refCoefs;//The coefficients of planes.
	vector<PointCloud<PointXYZI>::Ptr> m_restClusters;//The clusters after segmentation searching
	vector<PointIndices> m_restIndices;// The indices of clusters.
	PointCloud<PointXYZI>::Ptr m_candPins;//Candidate pins' point cloud.
	PointCloud<PointXYZI>::Ptr m_pinsPC;//Pins on the tyres, include positions(X,Y,Z) and length(I).
	vector<int> m_pinsID;// The head of pins' id in origin point cloud.
	PointCloud<PointXYZRGB>::Ptr m_rgbPC;//Point cloud with RGB.
	PointCloud<Normal>::Ptr m_pointNormals;//Point nomrals.

	map<int, vector<int>> m_clusterDic;//Key is the ID in refPlanes/refCoefs vector
									   //value is the ID vector of restClusters

	void InitCloudData();

	int FiltePins(Vector3d mineigenVector, vector<PointXYZI>& filted_pins);
	int FiltePins(vector<PointXYZI>& filted_pins);
public:
	PointCloud<PointXYZ>::Ptr GetOriginalPC();
	PointCloud<PointXYZI>::Ptr GetPinsPC();
	PointCloud<PointXYZRGB>::Ptr GetRGBPC();
	PointCloud<Normal>::Ptr GetPointNormals();

	//Get segementation references and clusters.
	void GetReferencePlanes(vector<PointCloud<PointXYZ>::Ptr> &out_ref);
	void GetReferenceCoefficients(vector<ModelCoefficients::Ptr> &out_ref);
	void GetRestClusters(vector<PointCloud<PointXYZI>::Ptr> &out_ref);

public:
	//Set parameters:
	void SetDownSampleRaius(float dsr);
	void SetNumberOfThreads(unsigned int nt);
	void SetDistanceThreshold(double dt);
	void SetNormalDistanceWeight(double ndw);
	void SetInlierRatio(double ir);
	void SetClusterTolerance(double ct);

public:
	int LoadTyrePC(string pcfile);
	/* Loading tyre point clouds from file, which contains in .ply or .pcd file.
	   Parameters:
	     pcfile(in): The input file directory of point clouds.
	*/

	int LoadTyrePC(PointCloud<PointXYZ>::Ptr in_cloud);
	/* Loading tyre point clouds from an exist cloud.
	   Parameters:
	     in_cloud(in): The input point cloud pointer.
	*/

	int FindPointNormals();
	/* Method to find out the point normals of point cloud(override).
	   Parameters:
	     None: default by neighbors=20, radius=0,folder=1, threads=2.
	*/

	int FindPointNormals(int neighbors, double radius, int folder, int threads);
	/* Method to find out the point normals of point cloud(override).
	   Parameters:
	     neighbors(in): K-neighbors of searching.
		 radius(in): Searching radius.
		 folder(in): Searching indices in point cloud, with 1/folder of points.
		 threads(in): The number of threads for multiple threads methods proprety.
	*/

	int FindPins(PointCloud<PointXYZI>::Ptr &out_pc);
	/* Searching pins on tyre surface(override):
	   Parameters:
	     out_pc(out): Out point cloud pointer with the length of pins.
	*/

	int FindPins(string pcfile, PointCloud<PointXYZI>::Ptr &out_pc);
	/* Searching pins on tyre surface(override):
	   Parameters:
	     pcfile(in): Input the point cloud from the directory.
	     out_pc(out): Out point cloud pointer with the length of pins.
	*/

	int FindPins(PointCloud<PointXYZ>::Ptr in_pc, PointCloud<PointXYZI>::Ptr &out_pc);
	/* Searching pins on tyre surface(override):
	   Parameters:
	     in_pc(in): Input point cloud pointer.
	     out_pc(out): Out point cloud pointer with the length of pins.
	*/

	int FindPins(PointCloud<PointXYZ>::Ptr in_pc, int neighbors, double radius, int folder, int threads, PointCloud<PointXYZI>::Ptr &out_pc);
	/* Searching pins on tyre surface(override):
	   Parameters:
	     in_pc(in): Input point cloud pointer.
		 neighbor(in): K-neighbors for searching normals.
		 radius(in): Searching radius for point normals method.
		 folder(in): Normal searching indices folder, which is the 1/folder of points.
		 threads(in): The number of threads, which will be used in the multiple threads normal searching method.
	     out_pc(out): Out point cloud pointer with the length of pins.
	*/

	int FindPinsBySegmentation(PointCloud<PointXYZ>::Ptr in_pc, PointCloud<PointXYZI>::Ptr &out_pc);

	int FindPins(char* p_pc, int length, vector<PinObject> & out_pc);
	/* Find pins by input a char stream
	   Parameters:
	     p_pc(in): Char pointer of the input point cloud.
		 length(in): The length of char stream.
		 out_pc(out): A list of PinObjects, which have position(x,y,z) and the length(len).
	*/

};

