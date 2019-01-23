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

	//Segementation Properties:
	m_downsampleradius = 300;
	m_numberofthreds = 2;
	m_distancethreshold = 300;
	m_normaldistanceweight = 0.5;
	m_inlierratio = 0.2;
	m_clustertolerance = 1000;

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

int TyrePointCloud::LoadTyrePC(PointCloud<PointXYZ>::Ptr in_cloud)
{
	InitCloudData();
	if (!in_cloud)
	{
		if (!m_originPC)
		{
			m_originPC.reset(::new PointCloud<PointXYZ>);
		}

		if (!m_rgbPC)
		{
			m_rgbPC.reset(::new PointCloud<PointXYZRGB>);
		}
		PointXYZ tmpPt;
		PointXYZRGB tmpRGB;
		size_t ii = 0;
		for ( ii= 0; ii < in_cloud->points.size(); ++ii)
		{
			tmpPt.x = in_cloud->points[ii].x;
			tmpPt.y = in_cloud->points[ii].y;
			tmpPt.z = in_cloud->points[ii].z;

			tmpRGB.x = in_cloud->points[ii].x;
			tmpRGB.y = in_cloud->points[ii].y;
			tmpRGB.z = in_cloud->points[ii].z;
			tmpRGB.rgb = 255;

			if (abs(tmpPt.x) > ACCURACY && abs(tmpPt.y) > ACCURACY && abs(tmpPt.z) > ACCURACY)
			{
				m_originPC->points.push_back(tmpPt);
			}
		}
		if (ii == 0)
		{
			return EMPTY_POINT;
		}
		else
		{
			return 0;
		}
	}
	else
	{
		return NULL_PC_PTR;
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
				break;
				/*if (hID == 0)
				{
					width = line_width;
				}
				if (line_width > 0 && line_width<width*2)
				{
					if (line_width > width)
					{
						width = line_width;
					}
				}
				hID++;*/
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
		radii.push_back(m_distancethreshold);
	}

	pcl::gpu::Octree::Queries queries_device;
	queries_device.upload(query_host);

	pcl::gpu::Octree::Radiuses radii_device;
	radii_device.upload(radii);

	const int max_answers = 200;

	// Output buffer on the device
	pcl::gpu::NeighborIndices result_device(queries_device.size(), max_answers);

	// Do the actual search
	octree_device->radiusSearch(queries_device, radii_device, max_answers, result_device);

	std::vector<int> sizes, data;
	result_device.sizes.download(sizes);
	result_device.data.download(data);

	pcl::gpu::DeviceArray<PointXYZ> normals;
	pcl::gpu::DeviceArray<PointXYZ> normals1;
	ne.setInputCloud(cloud_device);
	ne.setRadiusSearch(m_distancethreshold, max_answers);
	ne.setViewPoint(0.0, 0.0, 0.0);
	ne.compute(normals);
	ne.computeNormals(cloud_device, result_device, normals1);
	std::vector<PointXYZ> res_normals;
	std::vector<PointXYZ> res_normals1;
	normals.download(res_normals);
	normals1.download(res_normals1);

	Normal tmpNormal;
	m_gpuPtNormals.reset(::new PointCloud<Normal>);
	m_pointNormals.reset(::new PointCloud<Normal>);
	for (size_t ii = 0; ii < res_normals.size(); ii++)
	{
		tmpNormal.normal_x = res_normals[ii].x;
		tmpNormal.normal_y = res_normals[ii].y;
		tmpNormal.normal_z = res_normals[ii].z;
		m_gpuPtNormals->points.push_back(tmpNormal);

		tmpNormal.normal_x = res_normals1[ii].x;
		tmpNormal.normal_y = res_normals1[ii].y;
		tmpNormal.normal_z = res_normals1[ii].z;
		m_pointNormals->points.push_back(tmpNormal);
	}
	out_normal = m_gpuPtNormals;
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
	FiltPins(mineigenVector, pins);

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

/*int TyrePointCloud::FindPinsBySegmentation(PointCloud<PointXYZ>::Ptr in_pc, PointCloud<PointXYZI>::Ptr & out_pc)
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
		tree.reset(::new pcl::search::KdTree<PointXYZ>);
		cloud_plane.reset(::new PointCloud<PointXYZ>);
		coefficients.reset(::new ModelCoefficients);
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
	FiltePins(filted_pins);

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
}*/

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

	//pcl::gpu::Octree::PointCloud pt_device;
	//pt_device.upload(cloud_filtered->points);
	//oct->setCloud(pt_device);

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(::new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud_filtered);

	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	std::vector<pcl::PointIndices> cluster_indices;
	/*pcl::gpu::DeviceArray<PointXYZ> dev_pc;
	vector<PointXYZ> host_pc;
	for (size_t ii = 0; ii < cloud_filtered->points.size(); ii++)
	{
		host_pc.push_back(cloud_filtered->points[ii]);
	}
	dev_pc.upload(host_pc);*/
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
	string cur_str;
	bool b_pcData = false;
	int pos = 0;
	PointXYZ cur_pt;
	PointXYZRGB cur_rgb;
	float x, y, z;
	int r, g, b;
	int line_pos = 0;
	const char* t1 = " ";
	const char* t2 = "\n";
	m_inCloud = p_pc;
	while (pos<length)
	{
		if (!b_pcData)
		{
			if (*p_pc==*t1 || *p_pc==*t2)
			{
				if (cur_str!="")
				{
					cur_str.erase(0, cur_str.find_first_not_of(" "));
					if (cur_str == "end_header")
					{
						b_pcData = true;
					}
					cur_str = "";
				}
			}
			else
			{
				cur_str = cur_str + *p_pc;
			}
		}
		else
		{
			if (*p_pc == *t1 || *p_pc == *t2)
			{
				if (cur_str!="")
				{
					cur_str.erase(0, cur_str.find_first_not_of(" "));
					switch (line_pos)
					{
					case 0:
						x = stof(cur_str);
						break;
					case 1:
						y = stof(cur_str);
						break;
					case 2:
						z = stof(cur_str);
						break;
					case 3:
						r = stoi(cur_str);
						break;
					case 4:
						g = stoi(cur_str);
						break;
					case 5:
						b = stoi(cur_str);
						break;
					}
					cur_str = "";
					if (*p_pc==*t2)
					{
						line_pos = 0;
						cur_pt.x = x;
						cur_pt.y = y;
						cur_pt.z = z;
						m_originPC->points.push_back(cur_pt);

						cur_rgb.x = x;
						cur_rgb.y = y;
						cur_rgb.z = z;
						cur_rgb.r = r;
						cur_rgb.g = g;
						cur_rgb.b = b;
						m_rgbPC->points.push_back(cur_rgb);
					}
					else
					{
						line_pos++;
					}
				}
			}
			else
			{
				cur_str = cur_str + *p_pc;
			}
		}
		p_pc++;
		pos++;
	}

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

/*int ConvCharToValue(char * p_pc, pcl::PointCloud<PointXYZ>::Ptr& out_pt)
{
	if (!p_pc)
	{
		return EMPTY_CHAR_PTR;
	}
	else
	{
		string in_str = p_pc;
		stringstream oss;
		oss << in_str;
		string cur_str;
		thrust::host_vector<char*> str_host;
		char* tmpCharp;
		bool b_pcData = false;
		size_t len;
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
				len = cur_str.length();
				tmpCharp = (char*)malloc((len + 1) * sizeof(char));
				cur_str.copy(tmpCharp, len, 0);
				str_host.push_back(tmpCharp);
			}
			cur_str = "";
		}
		size_t lines_len = str_host.size();

		thrust::device_vector<char*> str_dev;
		thrust::copy(str_host.begin(), str_host.end(), str_dev.begin());

		pcl::gpu::DeviceArray<PointXYZRGB> pt_dev;
		pcl::PointCloud<PointXYZRGB>::Ptr pt_host;
		pt_host->points.resize(lines_len, PointXYZRGB());
		pt_dev.upload(pt_host->points);
		CharToValueDev(str_dev, pt_dev);
		pt_dev.download(pt_host->points);
	}

	return 0;
}*/


