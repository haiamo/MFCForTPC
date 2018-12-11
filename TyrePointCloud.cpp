//2018/12/3 TonyHE Create file
//2018/12/3 TonyHE Add class TyrePointCloud functions
//2018/12/5 TonyHE Modify the return list in FindPins function. 
//2018/12/9 TonyHE Realize the method of segementation searching for pins.
//2018/12/10 TonyHE Add set functions for SegSearching properties.

#include "stdafx.h"
#include "TyrePointCloud.h"

TyrePointCloud::TyrePointCloud()
{
	m_normalUserDefined = FALSE;

	//Segementation Properties:
	m_downsampleradius = 300;
	m_numberofthreds = 2;
	m_distancethreshold = 300;
	m_normaldistanceweight = 0.5;
	m_inlierratio = 0.2;
	m_clustertolerance = 1000;

	//Protected point clouds;
	m_originPC.reset(::new PointCloud<PointXYZ>);
	m_downsample.reset(::new PointCloud<PointXYZ>);
	m_segbase.reset(::new PointCloud<PointXYZ>);
	m_candPins.reset(::new PointCloud<PointXYZI>);
	m_rgbPC.reset(::new PointCloud<PointXYZRGB>);
	m_pointNormals.reset(::new PointCloud<Normal>);
}


TyrePointCloud::~TyrePointCloud()
{
	m_originPC.reset();
	m_downsample.reset();
	m_segbase.reset();
	m_candPins.reset();
	m_pinsPC.reset();
	m_rgbPC.reset();
	m_pointNormals.reset();
}

int TyrePointCloud::FiltePins(Vector3d mineigenVector, vector<PointXYZI>& filted_pins)
{
	PointXYZI tmpPt;
	Vector3d cur_vec, cur_proj, base_vec, base_proj;
	double angle;
	bool is_newVector = false;
	for (size_t ii = 0; ii < m_candPins->points.size(); ++ii)
	{
		tmpPt = m_candPins->points[ii];
		if (ii == 0)
		{
			filted_pins.push_back(tmpPt);
		}
		else
		{
			cur_vec = Vector3d(tmpPt.x, tmpPt.y, tmpPt.z);
			cur_proj = cur_vec - tmpPt.intensity*mineigenVector;
			is_newVector = true;
			for (vector<PointXYZI>::iterator jj = filted_pins.begin(); jj < filted_pins.end(); ++jj)
			{
				base_vec = Vector3d(jj->x, jj->y, jj->z);
				base_proj = base_vec - jj->intensity*mineigenVector;
				angle = acos((cur_proj.dot(base_proj)) / (cur_proj.norm()*base_proj.norm())) / M_PI * 180;
				if (angle < 1.0)
				{
					if (abs(tmpPt.intensity) > abs(jj->intensity))
					{
						*jj = tmpPt;
					}
					is_newVector = false;
				}
			}

			if (is_newVector)
			{
				filted_pins.push_back(tmpPt);
			}
		}
	}
	return 0;
}

PointCloud<PointXYZ>::Ptr TyrePointCloud::GetOriginalPC()
{
	return m_originPC;
}

PointCloud<PointXYZI>::Ptr TyrePointCloud::GetPinsPC()
{
	return m_pinsPC;
}

PointCloud<PointXYZRGB>::Ptr TyrePointCloud::GetRGBPC()
{
	return m_rgbPC;
}

PointCloud<Normal>::Ptr TyrePointCloud::GetPointNormals()
{
	return m_pointNormals;
}

void TyrePointCloud::SetDownSampleRaius(float dsr)
{
	m_downsampleradius = dsr;
}

void TyrePointCloud::SetNumberOfThreads(unsigned int nt)
{
	m_numberofthreds = nt;
}

void TyrePointCloud::SetDistanceThreshold(double dt)
{
	m_distancethreshold = dt;
}

void TyrePointCloud::SetNormalDistanceWeight(double ndw)
{
	m_normaldistanceweight = ndw;
}

void TyrePointCloud::SetInlierRatio(double ir)
{
	m_inlierratio = ir;
}

void TyrePointCloud::SetClusterTolerance(double ct)
{
	m_clustertolerance = ct;
}

int TyrePointCloud::LoadTyrePC(string pcfile)
{
	PointCloud<PointXYZ>::Ptr cloud(::new PointCloud<PointXYZ>);
	int f_error = -1;
	string file_type = "";

	if (0 == strcmp("", pcfile.data()))
	{
		return EMPTY_FILE_NAME;
	}
	else
	{
		file_type = pcfile.substr(pcfile.length() - 4, 4);
		if (0 == strcmp(file_type.data(), ".ply"))
		{
			f_error = pcl::io::loadPLYFile(pcfile, *cloud);
			if (-1 == f_error)
			{
				return LOAD_PLY_ERROR;
			}
			else
			{
				m_originPC = cloud;
				return 0;
			}
		}
		else if (0 == strcmp(file_type.data(), ".pcd"))
		{
			f_error = pcl::io::loadPCDFile(pcfile, *cloud);
			if (-1 == f_error)
			{
				return LOAD_PCD_ERROR;
			}
			else
			{
				m_originPC = cloud;
				return 0;
			}
		}
		else
		{
			return FILE_TYPE_ERROR;
		}
	}
}

int TyrePointCloud::LoadTyrePC(PointCloud<PointXYZ>::Ptr in_cloud)
{
	if (!in_cloud)
	{
		if (!m_originPC)
		{
			m_originPC.reset(::new PointCloud<PointXYZ>);
		}
		PointXYZ tmpPt;
		for (size_t ii = 0; ii < in_cloud->points.size(); ++ii)
		{
			tmpPt.x = in_cloud->points[ii].x;
			tmpPt.y = in_cloud->points[ii].y;
			tmpPt.z = in_cloud->points[ii].z;
			if (abs(tmpPt.x) > ACCURACY && abs(tmpPt.y) > ACCURACY && abs(tmpPt.z) > ACCURACY)
			{
				m_originPC->points.push_back(tmpPt);
			}
		}
		return 0;
	}
	else
	{
		return NULL_PC_PTR;
	}
}

int TyrePointCloud::FindPointNormals()
{
	int error = FindPointNormals(30, 0, 1, 2);
	m_normalUserDefined = FALSE;
	return error;
}

int TyrePointCloud::FindPointNormals(int neighbors, double radius, int folder, int threads)
{
	// Normal Estimation Process
	// Parameters preparation
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());//Create a null kdtree object.
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(::new pcl::PointCloud<pcl::Normal>);
	PointCloud<PointXYZ>::Ptr cloud = GetOriginalPC();

	//Estimating normal by multiple threads
	if (threads <= 0)
	{
		threads = 1;
	}
	ne.setNumberOfThreads(threads);
	ne.setInputCloud(cloud);

	//Transfer kdtree object to normal estimation object.
	//tree->setInputCloud(cloud);
	ne.setSearchMethod(tree);

	// Set searching neighbor radius or k-neighbors.
	if (radius <= 0 && neighbors>0)
	{
		ne.setKSearch(neighbors);
	}
	else if (radius > 0)
	{
		ne.setRadiusSearch(radius);
	}
	else
	{
		return NEGATIVE_R_K;
	}
	

	//Set searching indices of cloud points
	if (folder >= 2)
	{
		vector<int> indices(floor(cloud->points.size() / folder));
		for (int ii = 0; ii<indices.size(); ++ii)
		{
			indices[ii] = ii * int(folder);
		}
		IndicesPtr indicesptr(new vector<int>(indices));
		ne.setIndices(indicesptr);
	}

	//Searching normals
	ne.compute(*cloud_normals);
	m_pointNormals = cloud_normals;
	m_normalUserDefined = TRUE;
	return 0;
}

int TyrePointCloud::FindPins(PointCloud<PointXYZI>::Ptr &out_pc)
{
	int error =0;
	if (!m_normalUserDefined)
	{
		error = FindPointNormals();
		if (error < 0)
		{
			return error;
		}
	}
	
	error = FindPins(m_originPC, out_pc);
	return error;
}

int TyrePointCloud::FindPins(string pcfile, PointCloud<PointXYZI>::Ptr & out_pc)
{
	int error = 0;
	error = LoadTyrePC(pcfile);
	if (error < 0)
	{
		return error;
	}
	if (!m_normalUserDefined)
	{
		error = FindPointNormals();
		if (error < 0)
		{
			return error;
		}
	}
	
	error = FindPins(m_originPC, out_pc);
	return error;
}

int TyrePointCloud::FindPins(PointCloud<PointXYZ>::Ptr in_pc, PointCloud<PointXYZI>::Ptr & out_pc)
{
	int error = 0;
	if (!m_normalUserDefined)
	{
		error = FindPointNormals();
		if (error < 0)
		{
			return error;
		}
	}
	//Calculate the eigen vectors by PCA.
	Vector4f pcaCentroid;
	compute3DCentroid(*in_pc, pcaCentroid);
	Matrix3f covariance;
	computeCovarianceMatrixNormalized(*in_pc, pcaCentroid, covariance);
	SelfAdjointEigenSolver<Matrix3f> eigen_solver(covariance, ComputeEigenvectors);
	Matrix3f eigenVecotorsPCA = eigen_solver.eigenvectors();
	Vector3f eigenValuesPCA = eigen_solver.eigenvalues();

	//Compute pins' postions and lengths.
	PointCloud<Normal>::Ptr cur_normals = GetPointNormals();
	if (!m_rgbPC)
	{
		m_rgbPC.reset(::new PointCloud<PointXYZRGB>);
	}

	if (!m_candPins)
	{
		m_candPins.reset(::new PointCloud<PointXYZI>);
	}
	PointXYZRGB tmprgb;
	PointXYZI tmpi;
	Vector3d mineigenVector(eigenVecotorsPCA(0, 0), eigenVecotorsPCA(0, 1), eigenVecotorsPCA(0, 2));
	Vector3d normalPtr, pcaCent3d(pcaCentroid(0), pcaCentroid(1), pcaCentroid(2)), curpoint, curvector;
	vector<double> angles(cur_normals->points.size());
	vector<size_t> ppoinID;
	double cur_angle = 0.0, cur_depth = 0.0;
	for (size_t ii = 0; ii < cur_normals->points.size(); ++ii)
	{
		normalPtr = Vector3d(cur_normals->points[ii].normal_x, cur_normals->points[ii].normal_y, cur_normals->points[ii].normal_z);
		curpoint = Vector3d(in_pc->points[ii].x, in_pc->points[ii].y, in_pc->points[ii].z);
		curvector = curpoint - pcaCent3d;
		tmprgb.x = in_pc->points[ii].x;
		tmprgb.y = in_pc->points[ii].y;
		tmprgb.z = in_pc->points[ii].z;

		if (normalPtr.norm() < ACCURACY || _isnan(normalPtr[0]) || _isnan(normalPtr[1]) || _isnan(normalPtr[2]))
		{
			continue;
		}
		cur_angle = acos((normalPtr.dot(mineigenVector)) / (normalPtr.norm()*mineigenVector.norm())) / M_PI * 180;
		angles[ii] = cur_angle;
		if (cur_angle > 80 && cur_angle < 100)
		{
			ppoinID.push_back(ii);
			tmprgb.r = 255;
			tmprgb.g = 0;
			tmprgb.b = 0;
			cur_depth = curvector.dot(mineigenVector);
			if (cur_depth<0)
			{
				tmpi.x = in_pc->points[ii].x;
				tmpi.y = in_pc->points[ii].y;
				tmpi.z = in_pc->points[ii].z;
				tmpi.intensity = cur_depth;
				m_candPins->push_back(tmpi);
			}
		}
		else
		{
			tmprgb.r = 0;
			tmprgb.g = 0;
			tmprgb.b = 255;
			tmpi.intensity = 0.0;
		}
		m_rgbPC->points.push_back(tmprgb);
	}
	//out_pc = m_candPins;

	//Filting pins' postions and length.
	vector<PointXYZI> pins;
	FiltePins(mineigenVector, pins);

	if (!m_pinsPC)
	{
		m_pinsPC.reset(::new PointCloud<PointXYZI>);
	}

	for (vector<PointXYZI>::iterator ii = pins.begin(); ii < pins.end(); ++ii)
	{
		m_pinsPC->points.push_back(*ii);
	}
	out_pc = m_pinsPC;
	return 0;
}

int TyrePointCloud::FindPins(PointCloud<PointXYZ>::Ptr in_pc, int neighbors, double radius, int folder, int threads, PointCloud<PointXYZI>::Ptr & out_pc)
{
	FindPointNormals(neighbors, radius, folder, threads);
	FindPins(in_pc, out_pc);
	return 0;
}

int TyrePointCloud::FindPinsBySegmentation(PointCloud<PointXYZ>::Ptr in_pc, PointCloud<PointXYZI>::Ptr & out_pc)
{
	if (!in_pc)
	{
		return NULL_PC_PTR;
	}
	out_pc.reset(::new PointCloud<PointXYZI>);

	int error = 0;
	if (!m_normalUserDefined)
	{
		error = FindPointNormals();
		if (error < 0)
		{
			return error;
		}
	}
	//Calculate the eigen vectors by PCA.
	Vector4f pcaCentroid;
	compute3DCentroid(*in_pc, pcaCentroid);
	Matrix3f covariance;
	computeCovarianceMatrixNormalized(*in_pc, pcaCentroid, covariance);
	SelfAdjointEigenSolver<Matrix3f> eigen_solver(covariance, ComputeEigenvectors);
	Matrix3f eigenVecotorsPCA = eigen_solver.eigenvectors();
	Vector3f eigenValuesPCA = eigen_solver.eigenvalues();

	PointCloud<Normal>::Ptr cur_normals = GetPointNormals();
	if (!m_candPins)
	{
		m_candPins.reset(::new PointCloud<PointXYZI>);
	}
	Vector3d mineigenVector(eigenVecotorsPCA(0, 0), eigenVecotorsPCA(0, 1), eigenVecotorsPCA(0, 2));

	//Downsample the dataset, Needed?
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(::new pcl::PointCloud<pcl::PointXYZ>);
	
	if (m_downsampleradius > 0)
	{
		vg.setInputCloud(m_originPC);
		vg.setLeafSize(m_downsampleradius, m_downsampleradius, m_downsampleradius);
		vg.filter(*cloud_filtered);
	}
	else
	{
		cloud_filtered = m_originPC;
	}
	m_downsample = cloud_filtered;

	// Estimate point normals
	pcl::NormalEstimationOMP<PointXYZ, pcl::Normal> ne;
	PointCloud<Normal>::Ptr cur_normal(::new PointCloud<Normal>);
	pcl::search::KdTree<PointXYZ>::Ptr tree(::new pcl::search::KdTree<PointXYZ>());
	ne.setNumberOfThreads(m_numberofthreds);
	ne.setSearchMethod(tree);
	ne.setInputCloud(cloud_filtered);
	ne.setKSearch(50);
	if (!m_pointNormals)
	{
		m_pointNormals.reset(::new PointCloud<Normal>);
	}
	ne.compute(*cur_normal);
	m_pointNormals = cur_normal;


	// Create the segmentation object for the planar model and set all the parameters
	//pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::SACSegmentationFromNormals<PointXYZ, Normal>seg;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients(::new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(::new pcl::PointCloud<pcl::PointXYZ>());
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SacModel::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(10000);
	seg.setDistanceThreshold(m_distancethreshold);
	seg.setRadiusLimits(0, 100000);
	seg.setInputNormals(cur_normal);
	seg.setNormalDistanceWeight(m_normaldistanceweight);


	PointCloud<PointXYZ>::Ptr cloud_f(::new PointCloud<PointXYZ>);
	int i = 0, nr_points = (int)cloud_filtered->points.size();
	PointXYZ tmpPt;
	if (!m_segbase)
	{
		m_segbase.reset(::new PointCloud<PointXYZ>);
	}
	BOOL bNormalRenewed = TRUE;
	do//while (cloud_filtered->points.size() > m_inlierratio * nr_points)
	{
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud(cloud_filtered);
		if (!bNormalRenewed)
		{
			ne.setSearchMethod(tree);
			ne.setInputCloud(cloud_filtered);
			ne.compute(*cur_normal);
		}
		seg.setInputNormals(cur_normal);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() <0.5*nr_points)
		{
			if (seg.getModelType() == SACMODEL_PLANE)
			{
				seg.setModelType(SACMODEL_CYLINDER);
			}
			else if (seg.getModelType() == SACMODEL_CYLINDER)
			{
				seg.setModelType(SACMODEL_PLANE);
			}
		}

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(cloud_filtered);
		extract.setIndices(inliers);
		extract.setNegative(false);

		// Write the planar inliers to disk
		extract.filter(*cloud_plane);
		m_refPlanes.push_back(cloud_plane);
		for (size_t ii = 0; ii < cloud_plane->points.size(); ++ii)
		{
			tmpPt = cloud_plane->points[ii];
			m_segbase->points.push_back(tmpPt);
		}

		// Remove the planar inliers, extract the rest
		extract.setNegative(true);
		extract.filter(*cloud_f);
		cloud_filtered = cloud_f;
		cur_normal.reset(::new PointCloud<Normal>);
		tree.reset(::new pcl::search::KdTree<PointXYZ>);
		cloud_plane.reset(::new PointCloud<PointXYZ>);
		bNormalRenewed = FALSE;
	} while (cloud_filtered->points.size() > m_inlierratio * nr_points && cloud_plane->points.size() < 0.1*nr_points && m_refPlanes.size() < 15);
	
	// Creating the KdTree object for the search method of the extraction
	//pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(::new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(m_clustertolerance); 
	ec.setMinClusterSize(100);
	ec.setMaxClusterSize(25000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_filtered);
	ec.extract(cluster_indices);

	if (!cluster_indices.empty())
	{
		PointXYZI tmpPti;
		Vector3d curpoint, curvector, pcaCent3d(pcaCentroid(0), pcaCentroid(1), pcaCentroid(2));
		for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(::new pcl::PointCloud<pcl::PointXYZ>);
			for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
			{
				tmpPt = cloud_filtered->points[*pit];
				tmpPti.x = tmpPt.x;
				tmpPti.y = tmpPt.y;
				tmpPti.z = tmpPt.z;
				curpoint = Vector3d(tmpPt.x, tmpPt.y, tmpPt.z);
				curvector = curpoint - pcaCent3d;
				tmpPti.intensity = curvector.dot(mineigenVector);
				m_candPins->points.push_back(tmpPti);
			}
		}
	}

	vector<PointXYZI> filted_pins;
	FiltePins(mineigenVector, filted_pins);

	if (!m_pinsPC)
	{
		m_pinsPC.reset(::new PointCloud<PointXYZI>);
	}

	for (vector<PointXYZI>::iterator ii = filted_pins.begin(); ii < filted_pins.end(); ++ii)
	{
		m_pinsPC->points.push_back(*ii);
	}
	out_pc = m_pinsPC;
	return 0;
}
