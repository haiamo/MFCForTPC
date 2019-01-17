//2018/12/3 TonyHE Create file
//2018/12/3 TonyHE Add class TyrePointCloud functions
//2018/12/5 TonyHE Modify the return list in FindPins function. 
//2018/12/9 TonyHE Realize the method of segementation searching for pins.
//2018/12/10 TonyHE Add set functions for SegSearching properties.
//2018/12/12 TonyHE Realize the overloading funtion, FiltePins.
//2018/12/13 TonyHE Realize the fucntion of FindPin with char pointer and its length.

#include "stdafx.h"
#include "TyrePointCloud.h"

TyrePointCloud::TyrePointCloud()
{
	m_normalUserDefined = FALSE;

	//GPU Point Normal Searching Properties:
	m_searchradius = 500;
	m_maxneighboranswers = 200;

	//Segementation Properties:
	m_numberofthreds = 2;
	m_distancethreshold = 500;
	m_normaldistanceweight = 0.1;
	m_inlierratio = 0.2;
	m_clustertolerance = 300;

	InitCloudData();
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

void TyrePointCloud::InitCloudData()
{
	//Protected point clouds;
	m_originPC.reset(::new PointCloud<PointXYZ>);
	m_downsample.reset(::new PointCloud<PointXYZ>);
	m_segbase.reset(::new PointCloud<PointXYZ>);
	if(!m_refPlanes.empty())
		m_refPlanes.clear();
	if (!m_refCoefs.empty())
		m_refCoefs.clear();
	if(!m_restClusters.empty())
		m_restClusters.clear();
	m_candPins.reset(::new PointCloud<PointXYZI>);
	m_pinsPC.reset(::new PointCloud<PointXYZI>);
	m_rgbPC.reset(::new PointCloud<PointXYZRGB>);
	m_pointNormals.reset(::new PointCloud<Normal>);
	m_gpuPtNormals.reset(::new PointCloud<Normal>);
	if(!m_clusterDic.empty())
		m_clusterDic.clear();
}

int TyrePointCloud::FiltPins(Vector3d mineigenVector, vector<PointXYZI>& filted_pins)
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

int TyrePointCloud::FiltPins(vector<PointXYZI>& filted_pins)
{
	map<int, vector<int>>::iterator dic_it;
	vector<PointCloud<PointXYZ>::Ptr>::iterator ref_it=m_refPlanes.begin();
	vector<ModelCoefficients::Ptr>::iterator cof_it=m_refCoefs.begin();
	vector<PointCloud<PointXYZI>::Ptr>::iterator clu_it;
	vector<Vector3f> box_min, box_max;
	vector<Vector3f>::iterator boxmin_it, boxmax_it;
	Vector3f tmpMin(0.0f, 0.0f, 0.0f), tmpMax(0.0f, 0.0f, 0.0f), tmpCurPt(0.0f, 0.0f, 0.0f);
	//Finding the outer box of the reference planes or cylinders.
	for (ref_it = m_refPlanes.begin(); ref_it != m_refPlanes.end(); ref_it++)
	{
		for (size_t ii = 0; ii < (*ref_it)->points.size(); ii++)
		{
			tmpCurPt= Vector3f((*ref_it)->points[ii].x, (*ref_it)->points[ii].y, (*ref_it)->points[ii].z);
			if (ii == 0)
			{
				tmpMin = tmpCurPt;
				tmpMax = tmpCurPt;
				continue;
			}
			if ((tmpCurPt - tmpMin).cwiseSign() == Vector3f(-1.0, -1.0, -1.0))
			{
				tmpMin = tmpCurPt;
			}

			if ((tmpCurPt - tmpMax).cwiseSign() == Vector3f(1.0, 1.0, 1.0))
			{
				tmpMax = tmpCurPt;
			}
		}
		box_min.push_back(tmpMin);
		box_max.push_back(tmpMax);
	}

	float a, b, c, d;//Plane's coeficients.
	Vector3f proj_pt,cur_pt;
	vector<int> pt_in_plane(m_refPlanes.size(), 0), ini_pt_plane(m_refPlanes.size(), 0);;
	vector<int>::iterator pt_plane_it;
	vector<int>::iterator maxID;
	vector<int> tmp_cluster;
	//Making combination between surfaces and clusters.
	//Loop over the vector of rest point cloud clusters
	for (clu_it = m_restClusters.begin(); clu_it != m_restClusters.end(); clu_it++)
	{
		//Loop over the points in the cluster
		for(size_t ii=0;ii<(*clu_it)->points.size();ii++)
		{
			cur_pt = Vector3f((*clu_it)->points[ii].x, (*clu_it)->points[ii].y, (*clu_it)->points[ii].z);
			boxmin_it = box_min.begin();
			boxmax_it = box_max.begin();
			pt_plane_it = pt_in_plane.begin();
			cof_it = m_refCoefs.begin();
			//Loop of reference planes and coefficients
			while (cof_it != m_refCoefs.end())
			{
				if ((*cof_it)->values.size() == 4)//Plane
				{
					tmpMin = *boxmin_it;
					tmpMax = *boxmax_it;
					a = (*cof_it)->values[0];
					b = (*cof_it)->values[1];
					c = (*cof_it)->values[2];
					d = (*cof_it)->values[3];
					proj_pt(0) = ((b*b + c*c)*cur_pt(0) - a*(b*cur_pt(1) + c*cur_pt(2) + d)) / (a*a + b*b + c*c);
					proj_pt(1) = b / a*(proj_pt(0) - cur_pt(0)) + cur_pt(1);
					proj_pt(2) = c / a*(proj_pt(0) - cur_pt(0)) + cur_pt(2);
					if ((proj_pt(0) > tmpMin(0) && proj_pt(1) > tmpMin(1) && proj_pt(2) > tmpMin(2)) && 
						(proj_pt(0) < tmpMax(0) && proj_pt(1) < tmpMax(1) && proj_pt(2) < tmpMax(2)))
					{
						(*pt_plane_it)++;
						(*clu_it)->points[ii].intensity = (proj_pt - cur_pt).norm();
						break;
					}
				}
				else if ((*cof_it)->values.size() == 7)//Cylinder
				{
				}
				pt_plane_it++;
				cof_it++;
				boxmin_it++;
				boxmax_it++;
			}

		}
		maxID = max_element(pt_in_plane.begin(), pt_in_plane.end());
		if (*maxID > (*clu_it)->points.size()*0.8)
		{
			dic_it = m_clusterDic.find(int(maxID - pt_in_plane.begin()));
			
			if (dic_it != m_clusterDic.end())
			{
				(dic_it->second).push_back(int(clu_it - m_restClusters.begin()));
			}
			else
			{
				tmp_cluster.clear();
				tmp_cluster.push_back(int(clu_it - m_restClusters.begin()));
				m_clusterDic.insert(pair<int, vector<int>>(int(maxID - pt_in_plane.begin()), tmp_cluster));
			}
		}
		pt_in_plane = ini_pt_plane;
	}

	//Calculate the cluster distances to specific surfaces.
	PointXYZI tmpPin;
	float len = 0.0f;
	int ptID = 0;
	PointCloud<PointXYZI>::Ptr p_cluster;
	PointIndices cur_indices;
	int pinHeadID = 0;
	//Loop of map between the plane indices and cluster indices.
	for (dic_it = m_clusterDic.begin(); dic_it != m_clusterDic.end(); dic_it++)
	{
		cof_it = m_refCoefs.begin() + dic_it->first;
		//Loop through the vector of cluster onto the binding surface.
		for (int ii = 0; ii < dic_it->second.size(); ii++)
		{
			ptID = dic_it->second[ii];
			p_cluster = m_restClusters[ptID];
			cur_indices = m_restIndices[ptID];
			//Loop of the points in the cluster onto the surface.
			for (size_t jj = 0; jj < p_cluster->points.size(); jj++)
			{
				if (jj == 0)
				{
					len = p_cluster->points[jj].intensity;
					tmpPin = p_cluster->points[jj];
					pinHeadID = cur_indices.indices[jj];
				}
				else
				{
					if (p_cluster->points[jj].intensity > len)
					{
						len = p_cluster->points[jj].intensity;
						tmpPin = p_cluster->points[jj];
						pinHeadID = cur_indices.indices[jj];
					}
				}
			}
			filted_pins.push_back(tmpPin);
			m_pinsID.push_back(pinHeadID);
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

PointCloud<Normal>::Ptr TyrePointCloud::GetGPUPointNormals()
{
	return m_gpuPtNormals;
}

void TyrePointCloud::GetReferencePlanes(vector<PointCloud<PointXYZ>::Ptr>& out_ref)
{
	out_ref = m_refPlanes;
}

void TyrePointCloud::GetReferenceCoefficients(vector<ModelCoefficients::Ptr>& out_ref)
{
	out_ref = m_refCoefs;
}

void TyrePointCloud::GetRestClusters(vector<PointCloud<PointXYZI>::Ptr>& out_ref)
{
	out_ref = m_restClusters;
}

void TyrePointCloud::SetSearchRadius(float radius)
{
	m_searchradius = radius;
}

void TyrePointCloud::SetMaxNeighborAnswers(int maxans)
{
	m_maxneighboranswers = maxans;
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
	InitCloudData();
	PointCloud<PointXYZRGB>::Ptr cloudrgb(::new PointCloud<PointXYZRGB>);
	PointCloud<PointXYZ>::Ptr cloud(::new PointCloud<PointXYZ>);
	int f_error = -1;
	string file_type = "";
	m_originPC.reset(::new PointCloud<PointXYZ>);
	m_rgbPC.reset(::new PointCloud<PointXYZRGB>);

	if (0 == strcmp("", pcfile.data()))
	{
		return EMPTY_FILE_NAME;
	}
	else
	{
		file_type = pcfile.substr(pcfile.length() - 4, 4);
		if (0 == strcmp(file_type.data(), ".ply"))
		{
			f_error = pcl::io::loadPLYFile(pcfile, *cloudrgb);
			if (-1 == f_error)
			{
				return LOAD_PLY_ERROR;
			}
			else
			{
				if (cloudrgb->points.size() > 0)
				{
					m_rgbPC = cloudrgb;
					PointXYZ tmpPC;
					for (size_t ii = 0; ii < m_rgbPC->points.size(); ii++)
					{
						tmpPC.x = m_rgbPC->points[ii].x;
						tmpPC.y = m_rgbPC->points[ii].y;
						tmpPC.z = m_rgbPC->points[ii].z;
						m_originPC->points.push_back(tmpPC);
					}
					return 0;
				}
				else
				{
					f_error = pcl::io::loadPLYFile(pcfile, *cloud);
					if (-1 == f_error)
					{
						return LOAD_PLY_ERROR;
					}
					else
					{
						if (cloud->points.size() > 0)
						{
							PointXYZRGB tmpRGB;
							m_originPC = cloud;
							for (size_t ii = 0; ii < m_originPC->points.size(); ii++)
							{
								tmpRGB.x = m_originPC->points[ii].x;
								tmpRGB.y = m_originPC->points[ii].y;
								tmpRGB.z = m_originPC->points[ii].z;
								tmpRGB.rgb = 255;
								m_rgbPC->points.push_back(tmpRGB);
							}
							return 0;
						}
						else
						{
							return EMPTY_POINT;
						}
					}
				}
			}
		}
		else
		{
			return FILE_TYPE_ERROR;
		}
	}
}

int TyrePointCloud::LoadTyrePC(char * p_pc, int length)
{
	if (!p_pc)
	{
		return EMPTY_CHAR_PTR;
	}
	else
	{
		m_originPC.reset(::new pcl::PointCloud<PointXYZ>);
		m_rgbPC.reset(::new pcl::PointCloud<PointXYZRGB>);
		string in_str=p_pc;
		stringstream oss;
		oss << in_str;
		string cur_str;
		size_t line_width = 0,width=0,height = 0,hID=0;
		bool b_pcData = false;
		char** chars;
		while (getline(oss,cur_str))
		{
			if (!b_pcData)
			{
				if (cur_str == "end_header")
				{
					b_pcData = true;
				}
				if (cur_str.find("element vertex") != std::string::npos)
				{
					cur_str.erase(cur_str.begin(), cur_str.begin()+14);
					size_t index = 0;
					if (!cur_str.empty())
					{
						while ((index = cur_str.find(' ', index)) != string::npos)
						{
							cur_str.erase(index, 1);
						}
					}
					height = stoull(cur_str);
				}
			}
			else
			{
				line_width = cur_str.length();
				width = line_width * 2;
			}
		}

		
		chars = (char**)malloc(sizeof(char*)*height);
		for (size_t ii = 0; ii < height; ii++)
		{
			chars[ii] = (char*)malloc(sizeof(char)*(width+1));
		}
		oss.clear();
		oss << in_str;
		b_pcData = false;
		hID = 0;
		while (getline(oss, cur_str))
		{
			if (!b_pcData)
			{
				if (cur_str == "end_header")
				{
					b_pcData = true;
				}
			}
			else
			{
				if (hID > height-1)
				{
					break;
				}
				line_width = cur_str.length();
				cur_str.copy(chars[hID], line_width, 0);
				for (size_t ii = line_width; ii < width; ii++)
				{
					chars[hID][ii] = ' ';
				}
				chars[hID][width] = '\0';
				hID++;
			}
		}

		float* x, *y, *z;
		x = (float*)malloc(sizeof(float)*height);
		y = (float*)malloc(sizeof(float)*height);
		z = (float*)malloc(sizeof(float)*height);

		GPUCharToValue(chars,x,y,z,height,width);
		PointXYZRGB tmprgb;
		for (size_t ii = 0; ii < height; ii++)
		{
			tmprgb.x = x[ii];
			tmprgb.y = y[ii];
			tmprgb.z = z[ii];
			tmprgb.r = 150;
			tmprgb.g = 150;
			tmprgb.b = 150;
			m_originPC->points.push_back(PointXYZ(tmprgb.x, tmprgb.y, tmprgb.z));
			m_rgbPC->points.push_back(tmprgb);
		}
		free(x);
		free(y);
		free(z);

		for (size_t ii = 0; ii < height; ii++)
		{
			free(chars[ii]);
		}
		free(chars);
	}
	return 0;
}

int TyrePointCloud::FindPointNormalsGPU(PointCloud<PointXYZ>::Ptr in_pc, pcl::gpu::Octree::Ptr &in_tree, PointCloud<Normal>::Ptr &out_normal)
{
	pcl::gpu::NormalEstimation ne;
	PointCloud<PointXYZ>::Ptr cloud = in_pc;
	pcl::gpu::Octree::PointCloud cloud_device;
	cloud_device.upload(cloud->points);

	pcl::gpu::Octree::Ptr octree_device(in_tree);
	octree_device->setCloud(cloud_device);
	octree_device->build();

	// Create two query points
	std::vector<pcl::PointXYZ> query_host;
	std::vector<float> radii;
	for (size_t ii = 0; ii < cloud->points.size(); ii++)
	{
		query_host.push_back(cloud->points[ii]);
		radii.push_back(m_searchradius);
	}

	pcl::gpu::Octree::Queries queries_device;
	queries_device.upload(query_host);

	pcl::gpu::Octree::Radiuses radii_device;
	radii_device.upload(radii);

	// Output buffer on the device
	pcl::gpu::NeighborIndices result_device(queries_device.size(), m_maxneighboranswers);

	// Do the actual search
	octree_device->radiusSearch(queries_device, radii_device, m_maxneighboranswers, result_device);

	std::vector<int> sizes, data;
	result_device.sizes.download(sizes);
	result_device.data.download(data);

	pcl::gpu::DeviceArray<PointXYZ> normals;
	ne.setInputCloud(cloud_device);
	ne.setRadiusSearch(m_searchradius, m_maxneighboranswers);
	ne.setViewPoint(0.0, 0.0, 0.0);
	out_normal = m_gpuPtNormals;
	return 0;
}

int TyrePointCloud::FindPinsBySegmentationGPU(PointCloud<PointXYZ>::Ptr in_pc, PointCloud<PointXYZI>::Ptr & out_pc)
{
	if (!in_pc)
	{
		return NULL_PC_PTR;
	}
	out_pc.reset(::new PointCloud<PointXYZI>);

	int error = 0;
	// Estimate point normals
	PointCloud<Normal>::Ptr cur_normal(::new PointCloud<Normal>);
	pcl::gpu::Octree::Ptr oct(new pcl::gpu::Octree());
	if (!m_normalUserDefined)
	{
		error = FindPointNormalsGPU(in_pc, oct, cur_normal);
		if (error < 0)
		{
			return error;
		}
		m_gpuPtNormals = cur_normal;
	}
	//Calculate the eigen vectors by PCA.
	Vector4f pcaCentroid;
	compute3DCentroid(*in_pc, pcaCentroid);
	Matrix3f covariance;
	computeCovarianceMatrixNormalized(*in_pc, pcaCentroid, covariance);
	SelfAdjointEigenSolver<Matrix3f> eigen_solver(covariance, ComputeEigenvectors);
	Matrix3f eigenVecotorsPCA = eigen_solver.eigenvectors();
	Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
	Vector3d mineigenVector(eigenVecotorsPCA(0, 0), eigenVecotorsPCA(0, 1), eigenVecotorsPCA(0, 2));

	//Downsample the dataset, Needed?
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered = m_originPC;

	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentationFromNormals<PointXYZ, Normal>seg;
	pcl::PointIndices::Ptr inliers(::new pcl::PointIndices);
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
	size_t i = 0, nr_points = cloud_filtered->points.size();
	PointXYZ tmpPt;
	if (!m_segbase)
	{
		m_segbase.reset(::new PointCloud<PointXYZ>);
	}
	BOOL bNormalRenewed = TRUE;
	do
	{
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud(cloud_filtered);
		if (!bNormalRenewed)
		{
			oct.reset(new pcl::gpu::Octree());
			FindPointNormalsGPU(cloud_filtered, oct, cur_normal);
		}
		seg.setInputNormals(cur_normal);
		seg.segment(*inliers, *coefficients);
		m_refCoefs.push_back(coefficients);
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
		cloud_plane.reset(::new PointCloud<PointXYZ>);
		coefficients.reset(::new ModelCoefficients);
		bNormalRenewed = FALSE;
	} while (cloud_filtered->points.size() > m_inlierratio * nr_points && cloud_plane->points.size() < 0.1*nr_points && m_refPlanes.size() < 15);

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(::new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud_filtered);

	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	std::vector<pcl::PointIndices> cluster_indices;
	ec.setClusterTolerance(m_clustertolerance);
	ec.setMinClusterSize(100);
	ec.setMaxClusterSize(25000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_filtered);
	ec.extract(cluster_indices);
	m_restIndices = cluster_indices;

	if (!m_candPins)
	{
		m_candPins.reset(::new PointCloud<PointXYZI>);
	}
	PointXYZI tmpPti;
	Vector3d curpoint, curvector, pcaCent3d(pcaCentroid(0), pcaCentroid(1), pcaCentroid(2));
	if (!cluster_indices.empty())
	{
		for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
		{
			pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster(::new pcl::PointCloud<pcl::PointXYZI>);
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
				cloud_cluster->points.push_back(tmpPti);
			}
			m_restClusters.push_back(cloud_cluster);
		}
	}

	//Filtering pins with their neighbors.
	vector<PointXYZI> filted_pins;
	//FiltePins(mineigenVector, filted_pins);
	FiltPins(filted_pins);

	if (!m_pinsPC)
	{
		m_pinsPC.reset(::new PointCloud<PointXYZI>);
	}

	KdTreeFLANN<PointXYZ> kdtree;
	kdtree.setInputCloud(m_originPC);
	int k = 15;
	vector<int> pointID(k);
	vector<float> pointSqrDis(k);
	for (vector<PointXYZI>::iterator ii = filted_pins.begin(); ii < filted_pins.end(); ++ii)
	{
		m_pinsPC->points.push_back(*ii);
		tmpPt.x = ii->x;
		tmpPt.y = ii->y;
		tmpPt.z = ii->z;
		if (kdtree.nearestKSearch(tmpPt, k, pointID, pointSqrDis) > 0)
		{
			for (size_t jj = 0; jj < pointID.size(); jj++)
			{
				tmpPti.x = m_originPC->points[pointID[jj]].x;
				tmpPti.y = m_originPC->points[pointID[jj]].y;
				tmpPti.z = m_originPC->points[pointID[jj]].z;
				tmpPti.intensity = ii->intensity;
				m_pinsPC->points.push_back(tmpPti);

				m_rgbPC->points[pointID[jj]].r = 255;
				m_rgbPC->points[pointID[jj]].g = 255;
				m_rgbPC->points[pointID[jj]].b = 0;
			}
		}
	}
	out_pc = m_pinsPC;
	return 0;
}

int TyrePointCloud::FindPins(char * p_pc, int length, vector<PinObject>& out_pc)
{
	LoadTyrePC(p_pc, length);

	FindPinsBySegmentationGPU(m_originPC, m_pinsPC);

	PinObject tmpPXYZI;
	char* new_part = new char();
	for (size_t ii = 0; ii < m_pinsPC->points.size(); ii++)
	{
		tmpPXYZI.x = m_pinsPC->points[ii].x;
		tmpPXYZI.y = m_pinsPC->points[ii].y;
		tmpPXYZI.z = m_pinsPC->points[ii].z;
		tmpPXYZI.len = m_pinsPC->points[ii].intensity;
		out_pc.push_back(tmpPXYZI);
		sprintf(new_part, "%f %f %f 0 255 0\n",tmpPXYZI.x, tmpPXYZI.y, tmpPXYZI.z);
		strcat(m_inCloud, new_part);
	}
	p_pc = m_inCloud;
	return 0;
}
