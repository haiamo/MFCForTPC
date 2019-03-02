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

int TyrePointCloud::Get2DBaseLineBYRANSAC(pcl::PointCloud<PointXYZ>::Ptr in_pc, int min_inliers, int max_it, int order, double modelthreshold, int closepts)
{
	//Preparation for parameters
	//Point cloud for GPU
	double *xvals, *yvals, *dists, *paras;
	size_t pcsize = in_pc->points.size();
	int parasize = order+1;
	xvals = (double*)malloc(sizeof(double)*pcsize);
	yvals = (double*)malloc(sizeof(double)*pcsize);
	dists = (double*)malloc(sizeof(double)*pcsize);
	paras = (double*)malloc(sizeof(double) * parasize);
	pcl::PointCloud<PointXYZ>::iterator ptIt = in_pc->points.begin();
	size_t ptID = 0;
	while (ptIt < in_pc->points.end())
	{
		xvals[ptID] = ptIt->x;
		yvals[ptID] = ptIt->z;
		ptID++;
		ptIt++;
	}


	//Current this function will fit cubic line in 2D point cloud, which shows y=a*x^3+b*x^2+c*x+d
	vector<double> bestPara(parasize),candPara(parasize);
	double bestError = 0.0, candError = 0.0;
	vector<int> bestInliers, hypoInliers(min_inliers), candInliers(in_pc->points.size()), outliers, allIds(in_pc->points.size());
	vector<int>::iterator OutIdIt,bestIDit, hypoIDit;
	double bestErr = HUGE_VAL;
	int cur_it = 0, goodItTimes=0;
	for (int tt = 0; tt < in_pc->points.size(); tt++)
	{
		allIds[tt] = tt;
	}

	//Loop of outer iteration
	srand((unsigned int)time(0));
	while (cur_it < max_it && goodItTimes<max_it*0.1)
	{
		//min_inliers randomly selected points from input data
		
		for (int ii = 0; ii < min_inliers; ii++)
		{
			hypoInliers[ii] = rand() % in_pc->points.size();
		}
		sort(hypoInliers.begin(), hypoInliers.end());

		outliers = allIds;
		hypoIDit = hypoInliers.begin();
		OutIdIt =outliers.begin();
		while (OutIdIt < outliers.end())
		{
			if (hypoIDit < hypoInliers.end())
			{
				if (*OutIdIt == *hypoIDit)
				{
					OutIdIt = outliers.erase(OutIdIt);
					hypoIDit++;
				}
				else
				{
					OutIdIt++;
				}
			}
			else
			{
				break;
			}
		}
		//Get model parameters fitted to candInliers by Least Squares 
		PolyFit2D(hypoInliers, parasize-1, candPara);
		//Filtering inliers and outliers.
		candInliers.clear();
		candInliers.reserve(outliers.size());
		
		//Get distance by GPU
		for (int ii = 0; ii < candPara.size(); ii++)
		{
			paras[ii] = candPara[ii];
		}
		//paras[0] = candPara[0]; paras[1] = candPara[1]; paras[2] = candPara[2]; paras[3] = candPara[3];
		//GetPCDistance(xvals, yvals, pcsize, paras,4, dists);
		GetPCInterceptAsynch(xvals, yvals, pcsize, paras, parasize, dists);
		OutIdIt = outliers.begin();
		while (OutIdIt < outliers.end())
		{
			if (dists[*OutIdIt] < modelthreshold && dists[*OutIdIt] + modelthreshold > 0.0)
			{
				candInliers.push_back(*OutIdIt);
				//OutIdIt = outliers.erase(OutIdIt);
			}
			//else
			//{
				OutIdIt++;
			//}
		}

		/*OutIdIt = outliers.begin();
		double cur_dist = 0.0;
		while(OutIdIt < outliers.end())
		{
			//Calculating the distance between the current point and the model.
			cur_dist = PointToModelDistance(*OutIdIt, candPara, 3);
			if (cur_dist<modelthreshold)//Filtering inliers and outliers by the model.
			{
				candInliers.push_back(*OutIdIt);
				//OutIdIt = outliers.erase(OutIdIt);
			}
			//else
			//{
				++OutIdIt;
			//}
		}*/

		if (candInliers.size() > closepts)
		{
			//Get better model(candPara) by fitting candInliers and hypoInliers using Least Squares
			candInliers.insert(candInliers.end(), hypoInliers.begin(), hypoInliers.end());
			candError = PolyFit2D(candInliers, parasize-1, candPara);
			if (candError < bestError || goodItTimes==0)
			{
				bestPara = candPara;
				bestError = candError;
				bestInliers.assign(candInliers.begin(), candInliers.end());
			}
			goodItTimes++;
		}
		cur_it++;
	}//End of outer iteration

	bestIDit = bestInliers.begin();
	m_segbase->points.reserve(bestInliers.size());
	m_restPC->points.reserve(m_originPC->points.size()-bestInliers.size());
	size_t curID = 0;
	sort(bestInliers.begin(), bestInliers.end());
	while (bestIDit < bestInliers.end() && curID<m_originPC->points.size())
	{
		if (curID < *bestIDit)
		{
			m_restPC->points.push_back(m_originPC->points[curID]);
		}
		else if (curID == *bestIDit)
		{
			m_segbase->points.push_back(m_originPC->points[curID]);
			bestIDit++;
		}
		curID++;
	}

	free(xvals);
	free(yvals);
	free(dists);
	free(paras);
	xvals = NULL;
	yvals = NULL;
	dists = NULL;
	paras = NULL;
	return 0;
}

double TyrePointCloud::PolyFit2D(vector<int> in_pcID, int order, vector<double>& out_paras)
{
	Eigen::VectorXd xvals(in_pcID.size()), yvals(in_pcID.size());
	Eigen::MatrixXd A(in_pcID.size(),order+1);
	pcl::PointCloud<PointXYZ>::Ptr cloud = m_originPC;
	int curID = 0, rowi = 0;
	while (rowi < in_pcID.size())
	{
		curID = in_pcID[rowi];
		xvals(rowi) = cloud->points[curID].x;
		yvals(rowi) = cloud->points[curID].z;
		A(rowi, 0) = 1.0;
		for (int coli = 0; coli < order; coli++)
		{
			A(rowi, 1 + coli) = A(rowi, coli)*xvals(rowi);
		}
		rowi++;
	}
	auto Q = A.householderQr();
	auto Q1 = A.colPivHouseholderQr();
	Eigen::VectorXd beta = Q.solve(yvals);
	Eigen::VectorXd beta1 = Q1.solve(yvals);
	out_paras.clear();
	out_paras.reserve(beta.size());
	for (Eigen::Index ii = 0; ii < beta.size(); ii++)
	{
		out_paras.push_back(beta(ii));
	}
	Eigen::VectorXd errVec(in_pcID.size());
	errVec = A*beta - yvals;
	double errVal = errVec.norm();
	return errVal;
}

double TyrePointCloud::PointToModelDistance(int ptID, vector<double> model_paras, int order)
{
	double dist = 0.0, eps = 1e-5;
	pcl::PointCloud<PointXYZ>::Ptr cloud = m_originPC;
	Eigen::VectorXd iniPt(3), k_thPt(3), k_thF(3), k_thDeltX(3);
	iniPt(0) = cloud->points[ptID].x;
	iniPt(1) = cloud->points[ptID].z;
	iniPt(2) = 1.0;
	k_thPt = iniPt;
	double a = model_paras[3], b = model_paras[2], c = model_paras[1], d = model_paras[0];
	double g1 = 0.0, g2 = 0.0, detJ = 0.0;
	Eigen::MatrixXd k_thJ(3,3) , k_thJAdj(3, 3);
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
		//auto k_thQ = k_thJ.householderQr();
		//k_thDeltX = k_thQ.solve(k_thF);
		k_thDeltX = -1.0 / detJ*k_thJAdj*k_thF;
		k_thPt += k_thDeltX;
	} while ((abs(k_thF(0)) >= eps && abs(k_thF(1)) >= eps && abs(k_thF(2)) >= eps) &&
		(abs(k_thDeltX(0)) >= eps && abs(k_thDeltX(1)) >= eps && abs(k_thDeltX(2)) >= eps));
	dist = sqrt((k_thPt(0) - iniPt(0))*(k_thPt(0) - iniPt(0)) + (k_thPt(0) - iniPt(0))*(k_thPt(0) - iniPt(0)));
	return dist;
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

RangeImage::Ptr TyrePointCloud::GetRangeImage()
{
	return m_riPtr;
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

int TyrePointCloud::LoadTyrePC(string pcfile, float xLB, float xUB, float yStep, float zLB, float zUB, size_t width, size_t height)
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
			m_segmaxradius = sqrt((xUB - xLB)*(xUB - xLB) + yStep*height*yStep*height + (zUB - zLB)*(zUB - zLB));
			m_segminradius = min(abs(xUB - xLB) / width, yStep);
			if(input_file)
			{
				char* p;
				PointXYZ tmpXYZ;
				PointXYZRGB tmpRGB;
				size_t jj = 0;
				float cur_fl = 1.0f, step=0.0f;

				//Get a 32bit char, then convert it to float:
				while (input_file.peek() != EOF)
				{
					p = (char*)&cur_fl;
					input_file.read(p, sizeof(float));
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
						tmpXYZ.x = xLB + (jj % width) * (xUB - xLB) / width;
						tmpXYZ.y = step;
						tmpXYZ.z = cur_fl;
						if(tmpXYZ.x-xLB>8.0)
							cloud->points.push_back(tmpXYZ);
						/*tmpRGB.x = tmpXYZ.x;
						tmpRGB.y = tmpXYZ.y;
						tmpRGB.z = tmpXYZ.z;
						tmpRGB.r = 200;
						tmpRGB.g = 200;
						tmpRGB.b = 200;
						cloudrgb->points.push_back(tmpRGB);*/
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
	std::vector<pcl::PointXYZ> query_host(cloud->points.begin(),cloud->points.end());
	std::vector<float> radii(cloud->points.size(),m_distancethreshold);
	//for (size_t ii = 0; ii < cloud->points.size(); ii++)
	//{
		//query_host.push_back(cloud->points[ii]);
		//radii.push_back(m_distancethreshold);
	//}

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
	//extern "C" cudaError_t RANSACOnGPU(double* xvals, double* yvals, size_t pcsize, int maxIters, int minInliers, int parasize,
	//	double* &hst_hypox, double* &hst_hypoy, double* &As, double** &Qs, double** &taus, double** &Rs, double**& paras,
	//	double* &modelErr, double** &dists)
	cudaError_t cudaErr;
	size_t pcsize = in_pc->points.size();
	double *xvals, *yvals, *hst_hypox, *hst_hypoy, *As, *modelErr;
	double **Qs, **taus, **Rs, **paras, **dists;

	//Malloc spaces for data
	xvals = (double*)malloc(pcsize * sizeof(double));
	yvals = (double*)malloc(pcsize * sizeof(double));
	hst_hypox = (double*)malloc(maxIters *minInliers * sizeof(double));
	hst_hypoy = (double*)malloc(maxIters *minInliers * sizeof(double));
	As = (double*)malloc(sizeof(double)* maxIters * minInliers * paraSize);
	modelErr = (double*)malloc(sizeof(double) * maxIters);

	Qs = (double**)malloc(sizeof(double*)* maxIters);
	taus = (double**)malloc(sizeof(double*)* maxIters);
	Rs = (double**)malloc(sizeof(double*)*maxIters);
	paras = (double**)malloc(sizeof(double*) * maxIters);
	dists = (double**)malloc(sizeof(double*)*maxIters);

	for (int i = 0; i < maxIters; i++)
	{
		Qs[i] = (double*)malloc(sizeof(double) * minInliers * minInliers);
		memset(Qs[i], 0.0, sizeof(double)* minInliers * minInliers);
		taus[i] = (double*)malloc(sizeof(double) * paraSize);
		memset(taus[i], 0.0, sizeof(double)* paraSize);
		Rs[i] = (double*)malloc(sizeof(double)*minInliers*paraSize);
		memset(Rs[i], 0.0, sizeof(double)* minInliers * paraSize);
		paras[i] = (double*)malloc(sizeof(double)*paraSize);
		memset(paras[i], 0.0, sizeof(double)*paraSize);
		dists[i] = (double*)malloc(sizeof(double)*pcsize);
		memset(dists[0], 0.0, sizeof(double)*pcsize);
	}

	//Set input parameter values
	for (size_t ii = 0; ii < pcsize; ii++)
	{
		xvals[ii] = in_pc->points[ii].x;
		yvals[ii] = in_pc->points[ii].z;
	}

	cudaErr = RANSACOnGPU(xvals, yvals, pcsize, maxIters, minInliers, paraSize,
		hst_hypox, hst_hypoy, As, Qs, taus, Rs, paras, modelErr, dists);

	int bestIt = 0;
	double bestErr = modelErr[0];
	for (int ii = 1; ii < maxIters; ii++)
	{
		if (bestErr - modelErr[ii] > 1e-5)
		{
			bestErr = modelErr[ii];
			bestIt = ii;
		}
	}

	PointXYZ tmpPt;
	double* distList = *(dists + bestIt);
	for (size_t ii = 0; ii < pcsize; ii++)
	{
		tmpPt = in_pc->points[ii];
		if ((distList[ii] > 0 && distList[ii] < UTh) || (distList[ii] <= 0 && distList[ii] > LTh))
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
	if (NULL !=hst_hypox)
	{
		free(hst_hypox);
		hst_hypox = NULL;
	}
	if (NULL != hst_hypoy)
	{
		free(hst_hypoy);
		hst_hypoy = NULL;
	}
	if (NULL != As)
	{
		free(As);
		As = NULL;
	}
	if (NULL != modelErr)
	{
		free(modelErr);
		modelErr = NULL;
	}

	for (int i = 0; i < maxIters; i++)
	{
		if (NULL != Qs[i])
		{
			free(Qs[i]);
		}
		if (NULL != taus[i])
		{
			free(taus[i]);
		}
		if (NULL != Rs[i])
		{
			free(Rs[i]);
		}
		if (NULL != paras[i])
		{
			free(paras[i]);
		}
		if (NULL != dists[i])
		{
			free(dists[i]);
		}
	}
	free(Qs);
	Qs = NULL;
	free(taus);
	taus = NULL;
	free(Rs);
	Rs = NULL;
	free(paras);
	paras = NULL;
	free(dists);
	dists = NULL;
	return 0;
}

int TyrePointCloud::CovToRangeImage(vector<float> SesPos, float AngRes, float maxAngWid,
	float maxAngHgt, float noiseLevel, float minRange,
	int borderSize)
{
	PointCloud<PointXYZ>::Ptr pointCloud = GetOriginalPC();
	float angularResolution = (float)(AngRes * (M_PI / 180.0f));
	float maxAngleWidth = (float)(maxAngWid * (M_PI / 180.0f));
	float maxAngleHeight = (float)(maxAngHgt * (M_PI / 180.0f));
	//Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(SesPos[0], SesPos[1], SesPos[2]);
	pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;
	pcl::RangeImage::Ptr rangeImage;
	if (m_riPtr!=NULL)
	{
		rangeImage = m_riPtr;
	}
	else
	{
		m_riPtr.reset(::new RangeImage);
		rangeImage = m_riPtr;
	}
	//rangeImage->createFromPointCloud(*pointCloud, angularResolution, maxAngleWidth, maxAngleHeight, sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
	float viewPt[3] = { SesPos[0], SesPos[1], SesPos[2] };
	pcl::PointCloud<PointWithViewpoint>::Ptr ptrPtVP(::new pcl::PointCloud<PointWithViewpoint>);
	PointWithViewpoint tmpPtVP;
	pcl::PointCloud<PointXYZ>::iterator itPt = pointCloud->begin();
	for (; itPt < pointCloud->end(); itPt++)
	{
		tmpPtVP.x = itPt->x;
		tmpPtVP.y = itPt->y;
		tmpPtVP.z = itPt->z;
		tmpPtVP.vp_x = viewPt[0];
		tmpPtVP.vp_y = viewPt[1];
		tmpPtVP.vp_z = viewPt[2];
		ptrPtVP->points.push_back(tmpPtVP);
	}
	rangeImage->createFromPointCloudWithViewpoints(*ptrPtVP, angularResolution, maxAngleWidth, maxAngleHeight, coordinate_frame, noiseLevel, minRange, borderSize);
	m_riPtr = rangeImage;

	//float* ranges = rangeImage->getRangesArray();
	//unsigned char* rgb_image = pcl::visualization::FloatImageUtils::getVisualImage(ranges, rangeImage->width, rangeImage->height);
	//pcl::io::saveRgbPNGFile("saveRangeImageRGB.png", rgb_image, rangeImage->width, rangeImage->height);
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


