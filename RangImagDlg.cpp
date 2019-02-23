// RangImagDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "MFCForTPC.h"
#include "RangImagDlg.h"
#include "afxdialogex.h"


// RangImagDlg 对话框

IMPLEMENT_DYNAMIC(RangImagDlg, CDialogEx)

RangImagDlg::RangImagDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_RANGIMAGE, pParent)
{

}

RangImagDlg::~RangImagDlg()
{
}

void RangImagDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_SenPosX, m_edt_SenPosX);
	DDX_Control(pDX, IDC_SenPosY, m_edt_SenPosY);
	DDX_Control(pDX, IDC_SenPosZ, m_edt_SenPosZ);
	DDX_Control(pDX, IDC_NoiseLvl, m_edt_NoiseLvl);
	DDX_Control(pDX, IDC_MinRange, m_edt_MinRange);
	DDX_Control(pDX, IDC_MaxAngWid, m_edt_MaxAngWid);
	DDX_Control(pDX, IDC_MaxAngHgt, m_edt_MaxAngHgt);
	DDX_Control(pDX, IDC_BorderSize, m_edt_BorderSize);
	DDX_Control(pDX, IDC_AngRes, m_edt_AngRes);
}


BEGIN_MESSAGE_MAP(RangImagDlg, CDialogEx)
	ON_BN_CLICKED(IDOK, &RangImagDlg::OnBnClickedOk)
	ON_BN_CLICKED(IDCANCEL, &RangImagDlg::OnBnClickedCancel)
	ON_WM_KEYDOWN()
END_MESSAGE_MAP()


// RangImagDlg 消息处理程序


BOOL RangImagDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// TODO:  在此添加额外的初始化
	CString cur_val=_T("");
	cur_val.Format(_T("%.1f"), m_riProp.SenPos[0]);
	m_edt_SenPosX.SetWindowText(cur_val);

	cur_val.Format(_T("%.1f"), m_riProp.SenPos[1]);
	m_edt_SenPosY.SetWindowText(cur_val);

	cur_val.Format(_T("%.1f"), m_riProp.SenPos[2]);
	m_edt_SenPosZ.SetWindowText(cur_val);

	cur_val.Format(_T("%.1f"), m_riProp.AngRes);
	m_edt_AngRes.SetWindowText(cur_val);

	cur_val.Format(_T("%.1f"), m_riProp.MaxAngHgt);
	m_edt_MaxAngWid.SetWindowText(cur_val);

	cur_val.Format(_T("%.1f"), m_riProp.MaxAngHgt);
	m_edt_MaxAngHgt.SetWindowText(cur_val);

	cur_val.Format(_T("%.1f"), m_riProp.NoiseLvl);
	m_edt_NoiseLvl.SetWindowText(cur_val);

	cur_val.Format(_T("%.1f"), m_riProp.MinRange);
	m_edt_MinRange.SetWindowText(cur_val);

	cur_val.Format(_T("%d"), m_riProp.BorderSize);
	m_edt_BorderSize.SetWindowText(cur_val);
	/*m_edt_BorderSize.SetWindowText(cur_val);
	m_edt_SenPosX.SetWindowText(_T("0.0"));
	m_edt_SenPosY.SetWindowText(_T("0.0"));
	m_edt_SenPosZ.SetWindowText(_T("0.0"));
	m_edt_AngRes.SetWindowText(_T("1.0"));
	m_edt_MaxAngWid.SetWindowText(_T("360.0"));
	m_edt_MaxAngHgt.SetWindowText(_T("180.0"));
	m_edt_NoiseLvl.SetWindowText(_T("0.0"));
	m_edt_MinRange.SetWindowText(_T("0.0"));
	m_edt_BorderSize.SetWindowText(_T("1"));

	m_riProp.SenPos = vector<float>(3, 0.0);*/

	return TRUE;  // return TRUE unless you set the focus to a control
				  // 异常: OCX 属性页应返回 FALSE
}

void RangImagDlg::GetRIProp(RangeImageProperties & out_pro)
{
	out_pro = m_riProp;
}

void RangImagDlg::SetRIProp()
{
	CString cur_val;
	m_edt_SenPosX.GetWindowText(cur_val);
	m_riProp.SenPos[0] = stof(cur_val.GetBuffer());

	m_edt_SenPosY.GetWindowText(cur_val);
	m_riProp.SenPos[1] = stof(cur_val.GetBuffer());

	m_edt_SenPosZ.GetWindowText(cur_val);
	m_riProp.SenPos[2] = stof(cur_val.GetBuffer());

	m_edt_AngRes.GetWindowText(cur_val);
	m_riProp.AngRes = stof(cur_val.GetBuffer());

	m_edt_MaxAngWid.GetWindowText(cur_val);
	m_riProp.MaxAngWid = stof(cur_val.GetBuffer());

	m_edt_MaxAngHgt.GetWindowText(cur_val);
	m_riProp.MaxAngHgt = stof(cur_val.GetBuffer());

	m_edt_NoiseLvl.GetWindowText(cur_val);
	m_riProp.NoiseLvl = stof(cur_val.GetBuffer());

	m_edt_MinRange.GetWindowText(cur_val);
	m_riProp.MinRange = stof(cur_val.GetBuffer());

	m_edt_BorderSize.GetWindowText(cur_val);
	m_riProp.BorderSize = stof(cur_val.GetBuffer());
}

void RangImagDlg::SetRIProp(RangeImageProperties in_pro)
{
	m_riProp = in_pro;
}

void RangImagDlg::SetTPC(TyrePointCloud in_tpc)
{
	m_tpc = in_tpc;
}


void RangImagDlg::OnBnClickedOk()
{
	// TODO: 在此添加控件通知处理程序代码
	SetRIProp();
	RangeImageProperties curProp = m_riProp;
	//m_tpc.CovToRangeImage(curProp.SenPos, curProp.AngRes, curProp.MaxAngWid,
		//curProp.MaxAngHgt, curProp.NoiseLvl, curProp.MinRange, curProp.BorderSize);
	RangeImage::Ptr curRI = m_tpc.GetRangeImage();

	pcl::visualization::PCLVisualizer viewer("3D Viewer");
	viewer.setBackgroundColor(1, 1, 1); //背景
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> range_image_color_handler(curRI, 0, 0, 0);
	viewer.addPointCloud(curRI, range_image_color_handler, "range image");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "range image");
	viewer.addCoordinateSystem (1.0f, "global");
	//PointCloudColorHandlerCustom<PointType> point_cloud_color_handler (point_cloud_ptr, 150, 150, 150);
	//viewer.addPointCloud (point_cloud_ptr, point_cloud_color_handler, "original point cloud");
	viewer.initCameraParameters();
	//setViewerPose(viewer, range_image.getTransformationToWorldSystem ()); //PCL 1.6 出错

	pcl::visualization::RangeImageVisualizer RIView("RangeImage");
	RIView.showRangeImage(*curRI);

	Eigen::Affine3f sensorPose(Eigen::Affine3f::Identity()), tmpPose(Eigen::Affine3f::Identity());
	PointCloud<PointXYZ>::Ptr cloud = m_tpc.GetOriginalPC();
	vector<pcl::visualization::Camera> curCam;
	while (!viewer.wasStopped())//&& !RIView.wasStopped())
	{
		RIView.spinOnce();
		viewer.spinOnce();
		pcl_sleep(0.01);

		//sensorPose = (Eigen::Affine3f)Eigen::Translation3f(curProp.SenPos[0], curProp.SenPos[1], curProp.SenPos[2]);
		/*tmpPose = viewer.getViewerPose();
		if (!tmpPose.isApprox(sensorPose))
		{
			
			curRI = m_tpc.GetRangeImage();
			curRI->createFromPointCloud(*cloud, curProp.AngRes, curProp.MaxAngWid, curProp.MaxAngWid, tmpPose, pcl::RangeImage::LASER_FRAME, curProp.NoiseLvl, curProp.MinRange, curProp.BorderSize);
			RIView.showRangeImage(*curRI);
			sensorPose = tmpPose;
		}*/

		viewer.getCameras(curCam);
		if (abs(curProp.SenPos[0] - curCam[0].pos[0]) > 1e-5 && abs(curProp.SenPos[1] - curCam[0].pos[1]) > 1e-5 && abs(curProp.SenPos[2] - curCam[0].pos[2]) > 1e-5)
		{
			curProp.SenPos[0] = curCam[0].pos[0]*1000.0f;
			curProp.SenPos[1] = curCam[0].pos[1]*1000.0f;
			curProp.SenPos[2] = curCam[0].pos[2]*1000.0f;
			TRACE("Current Position: (%.1f, %.1f, %.1f)\n", curProp.SenPos[0], curProp.SenPos[1], curProp.SenPos[2]);
			//m_tpc.CovToRangeImage(curProp.SenPos, curProp.AngRes, curProp.MaxAngWid,
				//curProp.MaxAngHgt, curProp.NoiseLvl, curProp.MinRange, curProp.BorderSize);
			curRI = m_tpc.GetRangeImage();
			//RIView.close();
			RIView.showRangeImage(*curRI);
		}

	}
	//CDialogEx::OnOK();
}

RangeImageProperties::RangeImageProperties()
{
	SenPos = vector<float>(3, 0.0);
	AngRes = 1.0f;
	MaxAngWid = 360.0f;
	MaxAngHgt = 180.0f;
	NoiseLvl = 0.0f;
	MinRange = 0.0f;
	BorderSize = 1;
}


void RangImagDlg::OnBnClickedCancel()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnCancel();
}


void RangImagDlg::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值

	CDialogEx::OnKeyDown(nChar, nRepCnt, nFlags);
}


BOOL RangImagDlg::PreTranslateMessage(MSG* pMsg)
{
	// TODO: 在此添加专用代码和/或调用基类
	if (pMsg->message == WM_KEYDOWN && pMsg->wParam == VK_RETURN)
	{
		CString cur_val;
		int curCtrlID = GetFocus()->GetDlgCtrlID();
		if (curCtrlID == IDC_SenPosX || curCtrlID == IDC_SenPosY || curCtrlID == IDC_SenPosZ)//按下回车，如果当前焦点是在自己期望的控件上
		{
			m_edt_SenPosX.GetWindowText(cur_val);
			m_riProp.SenPos[0] = stof(cur_val.GetBuffer());
	
			m_edt_SenPosY.GetWindowText(cur_val);
			m_riProp.SenPos[1] = stof(cur_val.GetBuffer());

			m_edt_SenPosZ.GetWindowText(cur_val);
			m_riProp.SenPos[2] = stof(cur_val.GetBuffer());
			m_SenPosChanged = true;
		}
		return TRUE;
	}
	return CDialogEx::PreTranslateMessage(pMsg);
}
