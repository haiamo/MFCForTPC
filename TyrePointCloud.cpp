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
	m_searchradius = 0.09;
	m_maxneighboranswers = 200;

	//Segementation Properties:
	m_numberofthreds = 2;
	m_distancethreshold = 0.09;
	m_normaldistanceweight = 0.1;
	m_inlierratio = 0.2;
	m_clustertolerance = 0.09;
	m_segmaxradius = 1000;
	m_segminradius = 0;

	InitCloudData();
}


TyrePointCloud::~TyrePointCloud()
{
	m_originPC.reset();
	m_downsample.reset();
	m_segbase.reset();
	m_restPC.reset();
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
	m_restPC.reset(::new PointCloud<PointXYZ>);
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
	box_min.reserve(m_refPlanes.size());
	box_max.reserve(m_refPlanes.size());
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

int TyrePointCloud::SupervoxelClustering(pcl::PointCloud<PointXYZ>::Ptr in_pc, 
	std::map<uint32_t, pcl::Supervoxel<PointXYZ>::Ptr>& sv_cluster, 
	std::multimap<uint32_t, uint32_t>& sv_adj, 
	pcl::PointCloud<PointXYZL>::Ptr & lbl_pc,
	float voxel_res, float seed_res, float color_imp,
	float spatial_imp, float normal_imp)
{
	if (!in_pc)
	{
		return NULL_PC_PTR;
	}
	lbl_pc.reset(::new PointCloud<PointXYZL>);

	//Supervoxel generator
	pcl::SupervoxelClustering<pcl::PointXYZ> super(voxel_res, seed_res);
	//It is related the shape of point cloud.
	//if (disable_transform)
	super.setUseSingleCameraTransform(false);
	//Input PC and parameters.
	super.setInputCloud(in_pc);
	super.setColorImportance(color_imp);
	super.setSpatialImportance(spatial_imp);
	super.setNormalImportance(normal_imp);
	//Output supervoxel, which is a map between ID and Point Cloud pointer.
	std::map <uint32_t, pcl::Supervoxel<PointXYZ>::Ptr > supervoxel_clusters;
	super.extract(supervoxel_clusters);
	std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
	super.getSupervoxelAdjacency(supervoxel_adjacency);

	sv_cluster = supervoxel_clusters;
	sv_adj = supervoxel_adjacency;
	lbl_pc = super.getLabeledCloud();

	return 0;
}

void TyrePointCloud::SetOriginPC(PointCloud<PointXYZ>::Ptr in_pc)
{
	m_originPC = in_pc;
}

void TyrePointCloud::setOriginRGBPC(PointCloud<PointXYZRGB>::Ptr in_pc)
{
	m_rgbPC = in_pc;
}

PointCloud<PointXYZ>::Ptr TyrePointCloud::GetOriginalPC()
{
	return m_originPC;
}

PointCloud<PointXYZ>::Ptr TyrePointCloud::GetSegPC()
{
	return m_segbase;
}

PointCloud<PointXYZ>::Ptr TyrePointCloud::GetRestPC()
{
	return m_restPC;
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

int TyrePointCloud::LoadTyrePC(string pcfile, TPCProperty prop, float xBeg, float xEnd)
{
	float xLB, xUB, yStep, zLB, zUB, zStep,xStep,xOrigin,zOrigin;
	size_t width, height;
	int byteSize;
	string typeR, typeI;
	prop.GetAxisProp(&xLB, &xUB, &xStep,&xOrigin, 'x');
	prop.GetAxisProp(&zLB, &zUB, &zStep,&zOrigin, 'z');
	prop.GetAxisProp(NULL, NULL, &yStep,NULL, 'y');
	prop.GetWidthHeightBSize(&width, &height,&byteSize);
	prop.GetRIType(typeR, typeI);
	int errID = 0;
	string file_type = "";
	file_type = pcfile.substr(pcfile.length() - 4, 4);
	if (0 == strcmp(file_type.data(), ".dat"))
	{
		ifstream input_file;
		input_file.open(pcfile.data(), ios::binary);
		input_file.seekg(0, ios::end);
		height = input_file.tellg();
		height = height / (byteSize * width);
	}
	errID=LoadTyrePC(pcfile, xLB, xUB,  zLB, zUB,xStep,yStep,zStep,xOrigin,0.0f,zOrigin, width, height, xBeg, xEnd, typeR, typeI);
	return errID;
}

int TyrePointCloud::LoadTyrePC(string pcfile, float xLB, float xUB, float zLB, float zUB, float xStep, float yStep, float zStep,
	float xOrigin, float yOrigin, float zOrigin, size_t width, size_t height, float xBeg, float xEnd, string typeR, string typeI)
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
		else if (0 == strcmp(file_type.data(), ".dat"))
		{
			ifstream input_file;
			input_file.open(pcfile.data(), ios::binary);
			cloud->points.reserve(width * height);
			cloudrgb->points.reserve(width*height);
			//m_segmaxradius = sqrt((xUB - xLB)*(xUB - xLB) + yStep*height*yStep*height + (zUB - zLB)*(zUB - zLB));
			//m_segminradius = min(abs(xUB - xLB) / width, yStep);
			if(input_file)
			{
				char* p;
				PointXYZ tmpXYZ;
				PointXYZRGB tmpRGB;
				size_t jj = 0;
				float cur_fl = 1.0f, step=0.0f;
				unsigned int cur_ui = 0;
				int cur_i = 0;
				unsigned int byteR;
				string typeR_C;

				if ("BYTE" == typeR)
				{
					byteR = 1;
				}
				else if ("WORD" == typeR)
				{
					byteR = 2;
				}
				else if ("DWORD" == typeR || "INT" == typeR || "FLOAT" == typeR)
				{
					byteR = 4;
				}

				if ("BYTE" == typeR || "WORD" == typeR || "DWORD" == typeR)
				{
					p = (char*)&cur_ui;
				}
				else if ("INT" == typeR)
				{
					p = (char*)&cur_i;
				}
				else if ("FLOAT" == typeR)
				{
					p = (char*)&cur_fl;
				}

				while (input_file.peek() != EOF)
				{
					input_file.read(p, byteR);
					if ("BYTE" == typeR || "WORD" == typeR || "DWORD" == typeR)
					{
						cur_fl = cur_ui*1.0f;
					}
					else if ("INT" == typeR)
					{
						cur_fl = cur_i*1.0f;
					}
					if (jj % width == 0)
					{
						step += yStep;
					}

					if (cur_fl<zLB || cur_fl>zUB)
					{
						jj++;
						continue;
					}

					if (jj < width * height)
					{
						tmpXYZ.x = xOrigin + ((jj % width) - xLB) * xStep;
						tmpXYZ.y = step;
						tmpXYZ.z = zOrigin + (cur_fl - zLB) * zStep;
						if((jj % width)>=xBeg && (jj % width)<=xEnd)
							cloud->points.push_back(tmpXYZ);
					}
					else
					{
						break;
					}
					cur_fl = 0.0f;
					jj++;
				}
				m_originPC = cloud;
				m_rgbPC = cloudrgb;
				return 0;
			}
			else
			{
				return LOAD_CHAR_ERROR;
			}
		}
		else
		{
			return FILE_TYPE_ERROR;
		}
	}
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
	std::vector<pcl::PointXYZ> query_host(cloud->points.begin(),cloud->points.end());
	std::vector<float> radii(cloud->points.size(),m_distancethreshold);

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
	//ne.compute(normals);
	ne.computeNormals(cloud_device, result_device, normals);
	std::vector<PointXYZ> res_normals;
	normals.download(res_normals);

	m_gpuPtNormals.reset(::new PointCloud<Normal>);
	pcl::Normal tmpNormal;
	for (size_t ii = 0; ii < res_normals.size(); ii++)
	{
		tmpNormal.normal_x = res_normals[ii].x;
		tmpNormal.normal_y = res_normals[ii].y;
		tmpNormal.normal_z = res_normals[ii].z;
		m_gpuPtNormals->points.push_back(tmpNormal);
	}
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
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered = in_pc;

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
	seg.setRadiusLimits(m_segminradius, m_segmaxradius);
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
		m_segbase->points.insert(m_segbase->points.end(), cloud_plane->points.begin(), cloud_plane->points.end());

		// Remove the planar inliers, extract the rest
		extract.setNegative(true);
		extract.filter(*cloud_f);
		cloud_filtered = cloud_f;
		cur_normal.reset(::new PointCloud<Normal>);
		cloud_plane.reset(::new PointCloud<PointXYZ>);
		coefficients.reset(::new ModelCoefficients);
		bNormalRenewed = FALSE;
	} while (cloud_filtered->points.size() > m_inlierratio * nr_points && m_refPlanes.size() < 15);//&& cloud_plane->points.size() < m_inlierratio*nr_points

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(::new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud_filtered);

	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	std::vector<pcl::PointIndices> cluster_indices;
	ec.setClusterTolerance(m_clustertolerance);
	ec.setMinClusterSize(0);
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
		m_restClusters.clear();
		m_restClusters.reserve(cluster_indices.size());
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
	kdtree.setInputCloud(in_pc);
	int k = m_maxneighboranswers;
	vector<int> pointID(k);
	vector<float> pointSqrDis(k);
	m_pinsPC->points.clear();
	m_pinsPC->points.reserve(filted_pins.size()*k);
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

				m_rgbPC->points[pointID[jj]].r = 0;
				m_rgbPC->points[pointID[jj]].g = 0;
				m_rgbPC->points[pointID[jj]].b = 255;
			}
		}
	}
	out_pc = m_pinsPC;
	return 0;
}

int TyrePointCloud::FindCharsBySegmentationGPU(PointCloud<PointXYZ>::Ptr in_pc, PointCloud<PointXYZ>::Ptr & out_pc)
{
	if (!in_pc)
	{
		return NULL_PC_PTR;
	}
	/*vector<int> PtIDs;
	PtIDs.reserve(in_pc->points.size());
	for (int ii = 0; ii < in_pc->points.size(); ii++)
	{
		PtIDs.push_back(ii);
	}*/
	out_pc.reset(::new PointCloud<PointXYZ>);

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
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered = in_pc;

	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentationFromNormals<PointXYZ, Normal>seg;
	pcl::PointIndices::Ptr inliers(::new pcl::PointIndices), allinliers(::new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients(::new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(::new pcl::PointCloud<pcl::PointXYZ>());
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SacModel::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(10000);
	seg.setDistanceThreshold(m_distancethreshold);
	seg.setRadiusLimits(m_segminradius, m_segmaxradius);
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
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	//vector<int> NormIDs;
	do
	{
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud(cloud_filtered);
		if (!bNormalRenewed)
		{
			oct.reset(new pcl::gpu::Octree());
			FindPointNormalsGPU(cloud_filtered, oct, cur_normal);
			/*NormIDs.clear();
			NormIDs.reserve(PtIDs.size() - allinliers->indices.size());
			std::sort(allinliers->indices.begin(), allinliers->indices.end());
			set_difference(PtIDs.begin(),PtIDs.end(), allinliers->indices.begin(), allinliers->indices.end(), inserter(NormIDs,NormIDs.begin()));
			cur_normal->points.clear();
			for (vector<int>::iterator it = NormIDs.begin(); it < NormIDs.end(); it++)
			{
				cur_normal->points.push_back(m_gpuPtNormals->points.at((size_t)*it));
			}*/
		}
		seg.setInputNormals(cur_normal);
		seg.segment(*inliers, *coefficients);
		allinliers->indices.insert(allinliers->indices.end(), inliers->indices.begin(), inliers->indices.end());
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

		extract.setInputCloud(cloud_filtered);
		extract.setIndices(inliers);
		extract.setNegative(false);

		// Write the planar inliers to disk
		extract.filter(*cloud_plane);
		m_refPlanes.push_back(cloud_plane);
		m_segbase->points.insert(m_segbase->points.end(), cloud_plane->points.begin(), cloud_plane->points.end());

		// Remove the planar inliers, extract the rest
		extract.setNegative(true);
		extract.filter(*cloud_f);
		cloud_filtered = cloud_f;
		cur_normal.reset(::new PointCloud<Normal>);
		cloud_plane.reset(::new PointCloud<PointXYZ>);
		coefficients.reset(::new ModelCoefficients);
		bNormalRenewed = FALSE;
	} while (cloud_filtered->points.size() > m_inlierratio * nr_points && m_refPlanes.size() < 15);//&& cloud_plane->points.size() < m_inlierratio*nr_points

	extract.setInputCloud(in_pc);
	extract.setIndices(allinliers);
	extract.setNegative(true);
	extract.filter(*out_pc);

	pcl::PointXYZI tmpPtI;
	pcl::PointCloud<pcl::PointXYZI>::Ptr char_ptr(::new PointCloud<PointXYZI>);
	char_ptr->points.reserve(out_pc->points.size());
	for (pcl::PointCloud<PointXYZ>::iterator it = out_pc->points.begin(); it < out_pc->points.end(); it++)
	{
		tmpPtI.x = it->x;
		tmpPtI.y = it->y;
		tmpPtI.z = it->z;
		tmpPtI.intensity = 0.0f;
		char_ptr->points.push_back(tmpPtI);
	}
	m_restClusters.push_back(char_ptr);
	return 0;
}

int TyrePointCloud::FindCharsByLCCP(pcl::PointCloud<PointXYZ>::Ptr in_pc, pcl::PointCloud<PointXYZL>::Ptr & out_pc)
{
	if (!in_pc)
	{
		return NULL_PC_PTR;
	}
	out_pc.reset(::new PointCloud<PointXYZL>);

	//Supervoxel Clustering
	//Set parameters
	float voxel_resolution = 0.03f;
	float seed_resolution = 0.09f;
	float color_importance = 0.0f;
	float spatial_importance = 0.6f;
	float normal_importance = 1.0f;

	std::map <uint32_t, pcl::Supervoxel<PointXYZ>::Ptr > supervoxel_clusters;
	std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
	pcl::PointCloud<pcl::PointXYZL>::Ptr lccp_labeled_cloud;
	SupervoxelClustering(in_pc, supervoxel_clusters, supervoxel_adjacency, lccp_labeled_cloud);

	//Convex Conected Clustering
	//LCCP parameters:
	float concavity_tolerance_threshold = 0.2f, smoothness_threshold = 0.2f;
	uint32_t k_factor_arg = 10, min_segment_size_arg = 0;
	bool bool_use_sanity_criterion_arg = false, bool_use_smoothness_check_arg=false;

	//LCCP Segementation
	pcl::LCCPSegmentation<PointXYZ> seg;
	//Input Supervoxel results.
	seg.setInputSupervoxels(supervoxel_clusters, supervoxel_adjacency);
	//CC(Convexcity Criterion)
	seg.setConcavityToleranceThreshold(concavity_tolerance_threshold);
	//CC k neighbors
	seg.setKFactor(k_factor_arg);
	seg.setSmoothnessCheck(bool_use_smoothness_check_arg, voxel_resolution, seed_resolution, smoothness_threshold);
	//SC(Sanity Criterion)
	seg.setSanityCheck(bool_use_sanity_criterion_arg);
	seg.setMinSegmentSize(min_segment_size_arg);
	seg.segment();
	seg.relabelCloud(*lccp_labeled_cloud);
	out_pc = lccp_labeled_cloud;

	//Converting labeled cloud into cluster clouds.
	vector<uint32_t> labels;
	vector<uint32_t>::iterator lbl_it;
	labels.reserve(lccp_labeled_cloud->points.size());
	PointXYZI curPtI;
	pcl::PointCloud<PointXYZI>::Ptr curPCPtr;
	for (PointCloud<PointXYZL>::iterator it = lccp_labeled_cloud->points.begin(); it < lccp_labeled_cloud->points.end(); it++)
	{
		lbl_it = find(labels.begin(), labels.end(), it->label);
		if (lbl_it == labels.end() || labels.size()<1)//Con't find current label
		{
			labels.push_back(it->label);
			curPCPtr.reset(::new pcl::PointCloud<PointXYZI>);
			curPtI.x = it->x; 
			curPtI.y = it->y; 
			curPtI.z = it->z;
			curPCPtr->points.push_back(curPtI);
			m_restClusters.push_back(curPCPtr);
		}
		else//Current label exits.
		{
			curPCPtr = m_restClusters[distance(labels.begin(),lbl_it)];
			curPtI.x = it->x; 
			curPtI.y = it->y; 
			curPtI.z = it->z;
			curPCPtr->push_back(curPtI);
		}
	}

	return 0;
}

int TyrePointCloud::FindCharsByCPC(pcl::PointCloud<PointXYZ>::Ptr in_pc, pcl::PointCloud<PointXYZL>::Ptr & out_pc)
{
	if (!in_pc)
	{
		return NULL_PC_PTR;
	}
	out_pc.reset(::new PointCloud<PointXYZL>);

	//Supervoxel Clustering
	//Set parameters
	float voxel_resolution = 0.03f;
	float seed_resolution = 0.09f;
	float color_importance = 0.0f;
	float spatial_importance = 0.6f;
	float normal_importance = 1.0f;

	std::map <uint32_t, pcl::Supervoxel<PointXYZ>::Ptr > supervoxel_clusters;
	std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
	pcl::PointCloud<pcl::PointXYZL>::Ptr cpc_labeled_cloud;
	SupervoxelClustering(in_pc, supervoxel_clusters, supervoxel_adjacency, cpc_labeled_cloud);
	pcl::CPCSegmentation<PointXYZ> seg;
	seg.setInputSupervoxels(supervoxel_clusters, supervoxel_adjacency);
	//CPC Parameters:
	seg.setCutting(20, 0, 0.18, true, true, false);
	seg.setRANSACIterations(10000);
	seg.segment();
	seg.relabelCloud(*cpc_labeled_cloud);
	out_pc = cpc_labeled_cloud;
	//Converting labeled cloud into cluster clouds.
	vector<uint32_t> labels;
	vector<uint32_t>::iterator lbl_it;
	labels.reserve(cpc_labeled_cloud->points.size());
	PointXYZI curPtI;
	pcl::PointCloud<PointXYZI>::Ptr curPCPtr;
	for (PointCloud<PointXYZL>::iterator it = cpc_labeled_cloud->points.begin(); it < cpc_labeled_cloud->points.end(); it++)
	{
		lbl_it = find(labels.begin(), labels.end(), it->label);
		if (lbl_it == labels.end() || labels.size()<1)//Con't find current label
		{
			labels.push_back(it->label);
			curPCPtr.reset(::new pcl::PointCloud<PointXYZI>);
			curPtI.x = it->x;
			curPtI.y = it->y;
			curPtI.z = it->z;
			curPCPtr->points.push_back(curPtI);
			m_restClusters.push_back(curPCPtr);
		}
		else//Current label exits.
		{
			curPCPtr = m_restClusters[distance(labels.begin(), lbl_it)];
			curPtI.x = it->x;
			curPtI.y = it->y;
			curPtI.z = it->z;
			curPCPtr->push_back(curPtI);
		}
	}

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

int TyrePointCloud::FindCharsBy2DRANSACGPU(pcl::PointCloud<PointXYZ>::Ptr in_pc, int maxIters, int minInliers, int paraSize, double UTh, double LTh,
	pcl::PointCloud<PointXYZ>::Ptr & char_pc, pcl::PointCloud<PointXYZ>::Ptr & base_pc)
{
	cudaError_t cudaErr;
	size_t pcsize = in_pc->points.size();
	int resIters = 0;
	int *resInliers;
	double *xvals, *yvals, *paraList, *modelErr, *dists;

	//Malloc spaces for data
	resInliers = (int*)malloc(maxIters * sizeof(int));
	xvals = (double*)malloc(pcsize * sizeof(double));
	yvals = (double*)malloc(pcsize * sizeof(double));
	paraList = (double*)malloc(maxIters *paraSize * sizeof(double));
	modelErr = (double*)malloc(sizeof(double) * maxIters);
	dists = (double*)malloc(sizeof(double)*pcsize*maxIters);

	//Set input parameter values
	for (size_t ii = 0; ii < pcsize; ii++)
	{
		xvals[ii] = in_pc->points[ii].x;
		yvals[ii] = in_pc->points[ii].z;
	}

	cudaErr = RANSACOnGPU(xvals, yvals, pcsize, maxIters, minInliers, paraSize,UTh,LTh,
		paraList, resInliers, modelErr, dists, resIters);

	int bestIt = 0, maxInliers = resInliers[0];
	double bestErr = modelErr[0];
	for (int ii = 1; ii < resIters; ii++)
	{
		if (resInliers[ii] > maxInliers)
		{
			bestErr = modelErr[ii];
			maxInliers = resInliers[ii];
			bestIt = ii;
		}
		else if (resInliers[ii] == maxInliers && modelErr[ii] < bestErr)
		{
			bestErr = modelErr[ii];
			maxInliers = resInliers[ii];
			bestIt = ii;
		}
	}

	PointXYZ tmpPt;
	double* distList = dists + bestIt*pcsize;
	for (size_t ii = 0; ii < pcsize; ii++)
	{
		tmpPt = in_pc->points[ii];
		if ((distList[ii] > 0 && distList[ii] < UTh) || (distList[ii] <= 0 && distList[ii] > -LTh))
		{
			m_segbase->points.push_back(tmpPt);
		}
		else
		{
			m_restPC->points.push_back(tmpPt);
		}
	}
	char_pc = m_restPC;
	base_pc = m_segbase;

	//Free spaces
	if (NULL != resInliers)
	{
		free(resInliers);
		resInliers = NULL;
	}
	if (NULL != xvals)
	{
		free(xvals);
		xvals = NULL;
	}
	if (NULL != yvals)
	{
		free(yvals);
		yvals = NULL;
	}
	if (NULL != paraList)
	{
		free(paraList);
		paraList = NULL;
	}
	if (NULL != modelErr)
	{
		free(modelErr);
		modelErr = NULL;
	}
	if (NULL != dists)
	{
		free(dists);
		dists = NULL;
	}
	return 0;
}


