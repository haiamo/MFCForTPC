
// MFCForTPCDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "MFCForTPC.h"
#include "MFCForTPCDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMFCForTPCDlg 对话框



CMFCForTPCDlg::CMFCForTPCDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_MFCFORTPC_DIALOG, pParent)
	, m_RadioID(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMFCForTPCDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT_ClusterTolerance, m_edt_ClTol);
	DDX_Control(pDX, IDC_EDIT_DownSampleRadius, m_edt_DownSamR);
	DDX_Control(pDX, IDC_EDIT_NormalDistanceWeight, m_edt_NormDisWt);
	DDX_Control(pDX, IDC_EDIT_InilerRatio, m_edt_InlR);
	DDX_Control(pDX, IDC_EDIT_NumberOfThreads, m_edt_NumThds);
	DDX_Control(pDX, IDC_EDIT_DistanceThreshold, m_edt_DisThrhd);
	DDX_Control(pDX, IDC_STC_FilePath, m_stc_FlPth);
	DDX_Control(pDX, IDC_STC_Status, m_stc_St);
	DDX_Control(pDX, IDC_BTN_Run, m_btn_run);
	DDX_Control(pDX, IDC_BTN_SaveData, m_btn_savedata);
	DDX_Control(pDX, IDC_EDIT_xLB, m_edt_xLB);
	DDX_Control(pDX, IDC_EDIT_xUB, m_edt_xUB);
	DDX_Control(pDX, IDC_EDIT_yStep, m_edt_yStep);
	DDX_Control(pDX, IDC_EDIT_zLB, m_edt_zLB);
	DDX_Control(pDX, IDC_EDIT_zUB, m_edt_zUB);
	DDX_Control(pDX, IDC_EDIT_Width, m_edt_Width);
	DDX_Control(pDX, IDC_EDIT_Height, m_edt_Height);
	DDX_Control(pDX, IDC_EDIT_PCBeginID, m_edt_PCBeginID);
	DDX_Control(pDX, IDC_EDIT_PCEndID, m_edt_PCEndID);
}

BEGIN_MESSAGE_MAP(CMFCForTPCDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_Run, &CMFCForTPCDlg::OnBnClickedBtnRun)
	ON_BN_CLICKED(ID_BTN_Exit, &CMFCForTPCDlg::OnBnClickedBtnExit)
	ON_BN_CLICKED(IDC_BTN_SaveData, &CMFCForTPCDlg::OnBnClickedBtnSavedata)
	ON_BN_CLICKED(IDC_BUTTON2, &CMFCForTPCDlg::OnBnClickedButton2)
//	ON_BN_CLICKED(IDC_RanImg, &CMFCForTPCDlg::OnBnClickedRanimg)
	ON_BN_CLICKED(IDC_ShowPC, &CMFCForTPCDlg::OnBnClickedShowpc)
	ON_BN_CLICKED(IDC_RADIO_Pins, &CMFCForTPCDlg::OnBnClickedRadioPins)
	ON_BN_CLICKED(IDC_RADIO_Chars, &CMFCForTPCDlg::OnBnClickedRadioChars)
END_MESSAGE_MAP()


// CMFCForTPCDlg 消息处理程序

BOOL CMFCForTPCDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	m_edt_DownSamR.SetWindowText(_T("0.09"));
	m_edt_NumThds.SetWindowText(_T("2"));
	m_edt_DisThrhd.SetWindowText(_T("0.09"));
	m_edt_NormDisWt.SetWindowText(_T("0.1"));
	m_edt_InlR.SetWindowText(_T("0.1"));
	m_edt_ClTol.SetWindowText(_T("0.09"));

	m_edt_xLB.SetWindowText(_T("102.176"));
	m_edt_xUB.SetWindowText(_T("144.275"));
	m_edt_yStep.SetWindowText(_T("0.09"));
	m_edt_zLB.SetWindowText(_T("60.1541"));
	m_edt_zUB.SetWindowText(_T("144.275"));
	m_edt_Width.SetWindowText(_T("1536"));
	m_edt_Height.SetWindowText(_T("10000"));
	
	m_edt_PCBeginID.SetWindowText(_T("0"));
	m_edt_PCEndID.SetWindowText(_T("10000"));

	m_btn_savedata.EnableWindow(FALSE);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CMFCForTPCDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CMFCForTPCDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CMFCForTPCDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CMFCForTPCDlg::OnBnClickedBtnRun()
{
	m_bRanSeg = true;
	m_btn_run.EnableWindow(FALSE);
	m_btn_savedata.EnableWindow(FALSE);
	m_tpc.InitCloudData();
	CFileDialog fileopendlg(TRUE);
	CString filepath;
	if (fileopendlg.DoModal() == IDOK)
	{
		filepath = fileopendlg.GetPathName();
		m_stc_FlPth.SetWindowText(filepath);
	}
	else
	{
		m_stc_FlPth.SetWindowText(L"");
	}
	
	// Load point cloud file
	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	QueryPerformanceFrequency(&nfreq);
	PointCloud<PointXYZ>::Ptr cloud(::new PointCloud<PointXYZ>);//Create point cloud pointer.
	CString	cs_file;
	string pcfile = "";
	char* info_str = new char[1000];
	int str_len = 0;
	m_stc_FlPth.GetWindowTextW(cs_file);
	USES_CONVERSION;
	pcfile = CT2A(cs_file.GetBuffer());
	if (0 == strcmp("", pcfile.data()))
	{
		MessageBox(L"The file path is empty, please check again.", L"Load Info", MB_OK | MB_ICONERROR);
		m_stc_St.SetWindowText(L"Loading failed: empty file path");
		m_btn_run.EnableWindow(TRUE);
		m_btn_savedata.EnableWindow(TRUE);
		return;
	}
	else
	{
		int f_error = -1;
		CString xlb, xub, ystep, zlb, zub, width, height;
		float xmin, xmax, ys, zmin, zmax;
		size_t w, h;
		
		m_edt_xLB.GetWindowTextW(xlb);
		xmin = stof(xlb.GetBuffer());
		m_edt_xUB.GetWindowTextW(xub);
		xmax = stof(xub.GetBuffer());
		m_edt_yStep.GetWindowTextW(ystep);
		ys = stof(ystep.GetBuffer());
		m_edt_zLB.GetWindowTextW(zlb);
		zmin = stof(zlb.GetBuffer());
		m_edt_zUB.GetWindowTextW(zub);
		zmax = stof(zub.GetBuffer());
		m_edt_Width.GetWindowTextW(width);
		w = stoi(width.GetBuffer());
		m_edt_Height.GetWindowTextW(height);
		h = stoi(height.GetBuffer());

		QueryPerformanceCounter(&nst);
		f_error = m_tpc.LoadTyrePC(pcfile, xmin, xmax, ys, zmin, zmax, w, h);
		QueryPerformanceCounter(&nend);
		if (-1 == f_error)
		{
			MessageBox(L"Failed to load point cloud data, please try again!", L"LoadError", MB_OK | MB_ICONERROR);
			m_stc_St.SetWindowText(L"Loading failed: PCL function failed");
		}
		
		str_len = sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
		
		LPCWSTR info_wch = A2W(info_str);
		m_stc_St.SetWindowText(info_wch);
	}
	PointCloud<PointXYZ>::Ptr totalPC = m_tpc.GetOriginalPC();
	PointCloud<PointXYZRGB>::Ptr totalRGB = m_tpc.GetRGBPC();
	PointCloud<PointXYZRGB>::Ptr rgbCloud(::new pcl::PointCloud<PointXYZRGB>);
	CString beginID, endID;
	m_edt_PCBeginID.GetWindowTextW(beginID);
	m_edt_PCEndID.GetWindowTextW(endID);
	cloud->points.insert(cloud->points.begin(), totalPC->points.begin()+stoi(beginID.GetBuffer()), totalPC->points.begin() + stoi(endID.GetBuffer()));
	rgbCloud->points.insert(rgbCloud->points.begin(), totalRGB->points.begin() + stoi(beginID.GetBuffer()), totalRGB->points.begin() + stoi(endID.GetBuffer()));
	m_tpc.SetOriginPC(cloud);
	m_tpc.setOriginRGBPC(rgbCloud);

	SetSegParameters();
	char* tmp_str = new char[1000];
	if (m_RadioID == 1)
	{
		str_len = sprintf(info_str + str_len, "Starting pin searching, please wait...\n");
		LPCWSTR info_wstr = A2W(info_str);
		m_stc_St.SetWindowText(info_wstr);
		PointCloud<PointXYZI>::Ptr pins;
		QueryPerformanceCounter(&nst);
		m_tpc.FindPinsBySegmentationGPU(cloud, pins);
		QueryPerformanceCounter(&nend);
		memset(info_str, 0, sizeof(info_str)/sizeof(char));
		
		sprintf(tmp_str, "Searching pins costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
		info_str = strcat(info_str, tmp_str);
		info_wstr = A2W(info_str);
		m_stc_St.SetWindowText(info_wstr);
		size_t ii = 0;
		while (ii < pins->points.size() && ii < 5)
		{
			memset(tmp_str, 0, sizeof(tmp_str) / sizeof(char));
			sprintf(tmp_str, "%.2f, %.2f, %.2f, %.2f\n",pins->points[ii].x, pins->points[ii].y, pins->points[ii].z, pins->points[ii].intensity);
			info_str = strcat(info_str, tmp_str);
			ii++;
		}
		info_wstr = A2W(info_str);
		m_stc_St.SetWindowText(info_wstr);

	}
	else if (m_RadioID == 2)
	{
		str_len = sprintf(info_str + str_len, "Starting Characteristics searching, please wait...\n");
		LPCWSTR info_wstr = A2W(info_str);
		m_stc_St.SetWindowText(info_wstr);
		PointCloud<PointXYZ>::Ptr chars;
		QueryPerformanceCounter(&nst);
		m_tpc.FindCharsBySegmentationGPU(cloud, chars);
		QueryPerformanceCounter(&nend);
		memset(info_str, 0, sizeof(info_str) / sizeof(char));
		sprintf(tmp_str, "Searching chars costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
		info_str = strcat(info_str, tmp_str);
		info_wstr = A2W(info_str);
		m_stc_St.SetWindowText(info_wstr);
		pcl::io::savePLYFile("chars.ply", *chars);
	}

	m_btn_run.EnableWindow(TRUE);
	m_btn_savedata.EnableWindow(TRUE);

	//Pointers clearation.

	delete[] info_str;
	delete[] tmp_str;
	tmp_str = nullptr;
	info_str = nullptr;
	cloud.reset();
}


void CMFCForTPCDlg::OnBnClickedBtnExit()
{
	// TODO: 在此添加控件通知处理程序代码

	/*
	char* test = new char[1000];
	sprintf(test, "test %f", 100.2);
	delete[] test;
	
	//CDialogEx::OnOK();
	Vector3f tmp(10.0f, 11.0f, 12.0f);
	if ((tmp - Vector3f(10.0f-1e-6f, 10.0f, 11.0f)).cwiseSign() == Vector3f(1.0f, 1.0f, 1.0f))
	{
		TRACE("It is true.");
	}
	else
	{
		TRACE("It is false.");
	}*/
	CDialogEx::OnCancel();
}

int CMFCForTPCDlg::SaveXYZToPLYFile(vector<PointCloud<PointXYZ>::Ptr> in_pc, string ex_info)
{
	string savepath, ftype;
	GetPathAndType(savepath, ftype);
	int fe = 0, res = 0;
	vector<PointCloud<PointXYZ>::Ptr>::iterator it = in_pc.begin();
	for (it = in_pc.begin(); it < in_pc.end(); it++)
	{
		fe = pcl::io::savePLYFile(savepath + "_" + to_string(int(it - in_pc.begin())) + "_" + ex_info +"."+ ftype, **it);
		if (fe < 0)
		{
			res = res + fe;
		}
	}
	return res;
}

int CMFCForTPCDlg::SaveXYZIToPLYFile(vector<PointCloud<PointXYZI>::Ptr> in_pc, string ex_info)
{
	string savepath, ftype;
	GetPathAndType(savepath, ftype);
	int fe = 0, res = 0;
	vector<PointCloud<PointXYZI>::Ptr>::iterator it = in_pc.begin();
	for (it = in_pc.begin(); it < in_pc.end(); it++)
	{
		fe = pcl::io::savePLYFile(savepath + "_" + to_string(int(it - in_pc.begin())) + "_" + ex_info +"."+ ftype, **it);
		if (fe < 0)
		{
			res = res + fe;
		}
	}
	return res;
}

template<typename PointTPtr>
int CMFCForTPCDlg::SaveCloudToFile(vector<PointTPtr> p_inpc, string ex_info)
{
	string savepath, ftype;
	GetPathAndType(savepath, ftype);
	int fe = 0, res = 0;
	for (vector<PointTPtr>::iterator it = p_inpc.begin(); it < p_inpc.end(); it++)
	{
		bool bIsBin = false;
		if (ftype == "dat")
		{
			bIsBin = true;
			ftype = "ply";
		}
		fe = pcl::io::savePLYFile(savepath + "_" + to_string(int(it - p_inpc.begin())) + "_" + ex_info + "." + ftype, **it, bIsBin);
		if (fe < 0)
		{
			res += fe;
		}
	}
	return res;
}

template<typename PointTPtr>
int CMFCForTPCDlg::SaveCloudToFile(PointTPtr p_inpc, string ex_info)
{
	string savepath, ftype;
	GetPathAndType(savepath, ftype);
	int fe = 0;
	fe=pcl::io::savePLYFile(savepath + "_" + ex_info + "." + ftype, *p_inpc);
	return 0;
}

void CMFCForTPCDlg::SetSegParameters()
{
	CString cur_val;
	m_edt_DownSamR.GetWindowTextW(cur_val);
	//m_tpc.SetDownSampleRaius(stof(cur_val.GetBuffer()));
	m_tpc.SetSearchRadius(stof(cur_val.GetBuffer()));

	m_edt_NumThds.GetWindowTextW(cur_val);
	m_tpc.SetNumberOfThreads(stoi(cur_val.GetBuffer()));

	m_edt_DisThrhd.GetWindowTextW(cur_val);
	m_tpc.SetDistanceThreshold(stof(cur_val.GetBuffer()));

	m_edt_NormDisWt.GetWindowTextW(cur_val);
	m_tpc.SetNormalDistanceWeight(stof(cur_val.GetBuffer()));

	m_edt_InlR.GetWindowTextW(cur_val);
	m_tpc.SetInlierRatio(stof(cur_val.GetBuffer()));

	m_edt_ClTol.GetWindowTextW(cur_val);
	m_tpc.SetClusterTolerance(stof(cur_val.GetBuffer()));
}

int CMFCForTPCDlg::LoadPointCloud()
{
	CFileDialog fileopendlg(TRUE);
	CString filepath, prefilepath;
	m_stc_FlPth.GetWindowText(prefilepath);
	if (fileopendlg.DoModal() == IDOK)
	{
		filepath = fileopendlg.GetPathName();
		m_stc_FlPth.SetWindowText(filepath);
	}
	else
	{
		m_stc_FlPth.SetWindowText(L"");
	}

	if (prefilepath == filepath)
	{
		m_stc_St.SetWindowTextW(_T("The point cloud is the same."));
		return 0;
	}

	// Load point cloud file
	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	QueryPerformanceFrequency(&nfreq);
	PointCloud<PointXYZ>::Ptr cloud(::new PointCloud<PointXYZ>);//Create point cloud pointer.
	CString	cs_file;
	string pcfile = "";
	char* info_str = new char[1000];
	int str_len = 0;
	m_stc_FlPth.GetWindowTextW(cs_file);
	USES_CONVERSION;
	pcfile = CT2A(cs_file.GetBuffer());
	if (0 == strcmp("", pcfile.data()))
	{
		MessageBox(L"The file path is empty, please check again.", L"Load Info", MB_OK | MB_ICONERROR);
		m_stc_St.SetWindowText(L"Loading failed: empty file path");
	}
	else
	{
		int f_error = -1;

		QueryPerformanceCounter(&nst);
		f_error = m_tpc.LoadTyrePC(pcfile);
		QueryPerformanceCounter(&nend);
		if (-1 == f_error)
		{
			MessageBox(L"Failed to load point cloud data, please try again!", L"LoadError", MB_OK | MB_ICONERROR);
			m_stc_St.SetWindowText(L"Loading failed: PCL function failed");
		}

		str_len = sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);

		LPCWSTR info_wch = A2W(info_str);
		m_stc_St.SetWindowText(info_wch);
	}
	return 0;
}

void CMFCForTPCDlg::GetPathAndType(string & fpath, string & ftype)
{
	CString cs_file;
	m_stc_FlPth.GetWindowTextW(cs_file);

	string path, fullname, fname, savepath;
	size_t pathid = 0, ftypeid = 0;

	USES_CONVERSION;
	path = W2A(cs_file);
	pathid = path.find_last_of("\\");
	fullname = path.substr(pathid + 1);
	ftypeid = fullname.find_last_of(".");
	fname = fullname.substr(0, ftypeid);
	ftype = fullname.substr(ftypeid + 1);
	fpath = path.substr(0, pathid + 1) + fname;
}


void CMFCForTPCDlg::OnBnClickedBtnSavedata()
{
	// TODO: 在此添加控件通知处理程序代码
	vector<PointCloud<PointXYZ>::Ptr> refPlanes;
	m_tpc.GetReferencePlanes(refPlanes);
	SaveCloudToFile(refPlanes, "refPl");
	vector<PointCloud<PointXYZI>::Ptr> restClusters;
	m_tpc.GetRestClusters(restClusters);
	SaveCloudToFile(restClusters, "restCl");
	PointCloud<PointXYZI>::Ptr pins=m_tpc.GetPinsPC();
	vector<PointCloud<PointXYZI>::Ptr> vecPins;
	vecPins.push_back(pins);
	SaveCloudToFile(vecPins, "Pins");
	PointCloud<PointXYZRGB>::Ptr RGBCld = m_tpc.GetRGBPC();
	vector<PointCloud<PointXYZRGB>::Ptr> vecRGB;
	vecRGB.push_back(RGBCld);
	SaveCloudToFile(vecRGB, "RGB");
}


void CMFCForTPCDlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
	pcl::PointCloud<PointXYZ>::Ptr Cld_ptr(::new pcl::PointCloud<PointXYZ>);
	pcl::PointCloud<PointXYZ>::Ptr tmp_ptr(::new pcl::PointCloud<PointXYZ>);
	pcl::PointXYZ tmpPt;
	for (int ii = 0; ii < 3; ii++)
	{
		tmpPt.x = ii + 2;
		tmpPt.y = ii*ii;
		tmpPt.z = sin(ii);
		tmp_ptr->points.push_back(tmpPt);
		Cld_ptr->points.push_back(tmpPt);
	}
	Cld_ptr->points.insert(Cld_ptr->points.begin(), tmp_ptr->points.begin(), tmp_ptr->points.end());
	TRACE("The size of Cloud is %ld.\n", Cld_ptr->points.size());
	pcl::PointCloud<PointXYZ>::iterator it;
	int ii = 0;
	for (it=Cld_ptr->points.begin(); it < Cld_ptr->points.end(); it++)
	{
		TRACE("Cloud[%d] = (%f, %f, %f)\n", ii, it->x, it->y, it->z);
		ii++;
	}

	TRACE("The size of tmp is %ld.\n", tmp_ptr->points.size());
	ii = 0;
	for (it = tmp_ptr->points.begin(); it < tmp_ptr->points.end(); it++)
	{
		TRACE("tmp[%d] = (%f, %f, %f)\n", ii, it->x, it->y, it->z);
		ii++;
	}

	/*CFileDialog fileopendlg(TRUE);
	CString filepath;
	if (fileopendlg.DoModal() == IDOK)
	{
		filepath = fileopendlg.GetPathName();
	}

	long now1 = clock();
	std::ifstream t;
	int length;
	char* buffer;
	t.open(filepath);      // open input file  
	t.seekg(0, std::ios::end);    // go to the end  
	length = t.tellg();           // report location (this is the length)  
	t.seekg(0, std::ios::beg);    // go back to the beginning  
	buffer = new char[length];    // allocate memory for a buffer of appropriate dimension  
	t.read(buffer, length);       // read the whole file into the buffer  
	t.close();                    // close file handle  

	
	m_tpc.LoadTyrePC(buffer, length);
	char* gpustr = new char[100];
	sprintf(gpustr, "GPU running time: %.3f s\n", ((double)(clock() - now1)) / CLOCKS_PER_SEC);

	long now2 = clock();
	USES_CONVERSION;
	char* filename = T2A(filepath);
	m_tpc.LoadTyrePC(filename);
	char* cpustr = new char[100];
	sprintf(cpustr, "CPU runing time: %.3f s\n", ((double)(clock() - now2)) / CLOCKS_PER_SEC);
	strcat(gpustr, cpustr);
	MessageBoxA(this->GetSafeHwnd(), gpustr, "Run Result", 0);*/
	/*CUDA test
	CString cur_val;
	m_edt_DownSamR.GetWindowTextW(cur_val);
	int cursize = stoi(cur_val.GetBuffer());
	double* a = new double[cursize];
	double* b = new double[cursize];
	double* cg = new double[cursize];
	double* c = new double[cursize];
	//int a[cursize], b[cursize], c[cursize],cg[cursize];
	for (int i = 0; i < cursize; i++)
	{
		a[i] = i*1.0;
		b[i] = i*i*1.0;
	}

	long now1 = clock();//存储图像处理开始时间  
	runtest(a, b, cg, cursize);//调用显卡加速
	char* gpustr = new char[100];
	sprintf(gpustr, "GPU running time: %fms\n", ((double)(clock() - now1)) / CLOCKS_PER_SEC * 1000);//输出GPU处理时间

	long now2 = clock();//存储图像处理开始时间  
	int dif = 0;
	for (int i = 0; i < cursize; i++)
	{
		c[i] = a[i] *sin(b[i]);
		if (abs(cg[i] - c[i]) > 1e-5)
		{
			dif++;
		}
	}
	char* cpustr = new char[100];
	sprintf(cpustr, "CPU runing time: %fms\n", ((double)(clock() - now2)) / CLOCKS_PER_SEC * 1000);//输出GPU处理时间
	char* difstr = new char[50];
	sprintf(difstr, "They have %d different values.", dif);
	strcat(cpustr, difstr);
	strcat(gpustr, cpustr);
	MessageBoxA(this->GetSafeHwnd(), gpustr, "Run Result", 0);
	delete[] gpustr;
	delete[] cpustr;
	delete[] difstr;
	delete[] a;
	delete[] b;
	delete[] c;
	delete[] cg;
	gpustr = nullptr;
	cpustr = nullptr;
	difstr = nullptr;
	a = nullptr;
	b = nullptr;
	c = nullptr;
	cg = nullptr;*/

	/*USES_CONVERSION;
	CFileDialog fileopendlg(TRUE);
	CString filepath;
	bool b_loadfile = true;
	if (fileopendlg.DoModal() == IDOK)
	{
		filepath = fileopendlg.GetPathName();
		CString pre_file;
		m_stc_FlPth.GetWindowText(pre_file);
		m_stc_FlPth.SetWindowText(filepath);
		if (0 == strcmp(W2A(pre_file.GetBuffer()), W2A(filepath.GetBuffer())))
		{
			b_loadfile = false;
		}
	}
	else
	{
		m_stc_FlPth.SetWindowText(L"");
	}

	// Load point cloud file
	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	QueryPerformanceFrequency(&nfreq);
	PointCloud<PointXYZ>::Ptr cloud(::new PointCloud<PointXYZ>);//Create point cloud pointer.
	CString	cs_file;
	string pcfile = "";
	char* info_str = new char[1000];
	char* res_str = new char[1000];
	char* tmp_str = new char[100];
	LPCWSTR info_wstr;
	int str_len = 0;
	m_stc_FlPth.GetWindowTextW(cs_file);
	pcfile = CT2A(cs_file.GetBuffer());
	if (0 == strcmp("", pcfile.data()))
	{
		MessageBox(L"The file path is empty, please check again.", L"Load Info", MB_OK | MB_ICONERROR);
		m_stc_St.SetWindowText(L"Loading failed: empty file path");
	}
	else
	{
		
		if (b_loadfile)
		{
			int f_error = -1;

			QueryPerformanceCounter(&nst);
			f_error = m_tpc.LoadTyrePC(pcfile);
			QueryPerformanceCounter(&nend);
			if (-1 == f_error)
			{
				MessageBox(L"Failed to load point cloud data, please try again!", L"LoadError", MB_OK | MB_ICONERROR);
				m_stc_St.SetWindowText(L"Loading failed: PCL function failed");
			}

			sprintf(tmp_str, "Loading text costs %.3f seconds.\nStart normal searching, please wait...", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
			sprintf(res_str,"Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
			strcpy(info_str, tmp_str);
			info_wstr = A2W(info_str);
			m_stc_St.SetWindowText(info_wstr);
			strcpy(info_str, res_str);
		}
		else
		{
			sprintf(tmp_str, "Start normal searching, please wait...");
			strcpy(res_str, "");
			strcpy(info_str, tmp_str);
			info_wstr = A2W(info_str);
			m_stc_St.SetWindowText(info_wstr);
			strcpy(info_str, res_str);
		}


		SetSegParameters();
		PointCloud<Normal>::Ptr ptNormal;
		PointCloud<Normal>::Ptr gpuNormal;
		CString cur_val;
		m_edt_DisThrhd.GetWindowTextW(cur_val);

		/*long now1 = clock();
		m_tpc.FindPointNormals(0, stof(cur_val.GetBuffer()), 1, 4);
		sprintf(tmp_str, "CPU running time: %f s.\n", ((float)(clock() - now1)) / CLOCKS_PER_SEC);
		strcat(info_str, tmp_str);
		strcat(res_str, tmp_str);
		strcpy(tmp_str, "Normal searching by GPU, please wait...");
		strcat(info_str, tmp_str);
		info_wstr = A2W(info_str);
		m_stc_St.SetWindowText(info_wstr);
		strcpy(info_str, res_str);

		long now2 = clock();
		m_tpc.FindPointNormalsGPU();
		sprintf(tmp_str, "GPU runing time: %f s.\n", ((float)(clock() - now2)) / CLOCKS_PER_SEC);
		strcat(info_str, tmp_str);
		strcat(res_str, tmp_str);
		strcpy(tmp_str, "Start compearing data by CPU and GPU, please wait...");
		strcat(info_str, tmp_str);
		info_wstr = A2W(info_str);
		m_stc_St.SetWindowText(info_wstr);
		strcpy(info_str, res_str);
		
		ptNormal = m_tpc.GetPointNormals();
		gpuNormal = m_tpc.GetGPUPointNormals();
		size_t difSize = 0;
		
		Vector3d cpuN, gpuN;
		double angle = 0.0;
		if (ptNormal->points.size() == gpuNormal->points.size())
		{
			for (size_t ii = 0; ii < ptNormal->points.size(); ii++)
			{
				cpuN = Vector3d(ptNormal->points[ii].normal_x, ptNormal->points[ii].normal_y, ptNormal->points[ii].normal_z);
				gpuN = Vector3d(gpuNormal->points[ii].normal_x, gpuNormal->points[ii].normal_y, gpuNormal->points[ii].normal_z);
				angle= acos(cpuN.dot(gpuN) / (cpuN.norm()*gpuN.norm())) / M_PI * 180;
				if (angle>6.0f && angle<174.0f)
				{
					difSize++;
				}
			}
			sprintf(tmp_str, "There are %zd different elements between CPU and GPU normals.", difSize);
			strcat(info_str, tmp_str);
			info_wstr = A2W(info_str);
			m_stc_St.SetWindowText(info_wstr);
		}
		else
		{
			strcpy(tmp_str, "The normal lists by CPU and GPU have different sizes.");
			strcat(info_str, tmp_str);
			info_wstr = A2W(info_str);
			m_stc_St.SetWindowText(info_wstr);
		}
	}

	delete[] info_str;
	delete[] tmp_str;
	info_str = nullptr;
	tmp_str = nullptr;*/
}


/*void CMFCForTPCDlg::OnBnClickedRanimg()
{
	// TODO: 在此添加控件通知处理程序代码
	//RangeImageProperties curProp;
	LoadPointCloud();

	RangImagDlg RIDlg;
	RIDlg.SetRIProp(m_RIProp);
	RIDlg.SetTPC(m_tpc);
	RIDlg.DoModal();
	RIDlg.GetRIProp(curProp);
	m_RIProp = curProp;
	
	//m_tpc.CovToRangeImage(curProp.SenPos, curProp.AngRes, curProp.MaxAngWid,
		//curProp.MaxAngHgt, curProp.NoiseLvl, curProp.MinRange, curProp.BorderSize);
	//pcl::RangeImage::Ptr curRI = m_tpc.GetRangeImage();
	//pcl::visualization::RangeImageVisualizer curRIV("Tyre Range Image");
	//curRIV.showRangeImage(*curRI);
	//float* ranges = curRI->getRangesArray();
	//unsigned char* rgb_image = pcl::visualization::FloatImageUtils::getVisualImage(ranges, curRI->width, curRI->height);
	//pcl::io::saveRgbPNGFile("saveRangeImageRGB.png", rgb_image, curRI->width, curRI->height);
}*/


void CMFCForTPCDlg::OnBnClickedShowpc()
{
	// TODO: 在此添加控件通知处理程序代码
	CString xlb, xub, ystep, zlb, zub, width, height;
	float xmin, xmax, ys, zmin, zmax;
	size_t w, h;
	if (!m_bRanSeg)
	{
		CFileDialog fileopendlg(TRUE);
		CString filepath;
		if (fileopendlg.DoModal() == IDOK)
		{
			filepath = fileopendlg.GetPathName();
			m_stc_FlPth.SetWindowText(filepath);
		}
		else
		{
			m_stc_FlPth.SetWindowText(L"");
		}

		// Load point cloud file
		LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
		QueryPerformanceFrequency(&nfreq);
		
		CString	cs_file;
		string pcfile = "";
		char* info_str = new char[1000];
		int str_len = 0;
		m_stc_FlPth.GetWindowTextW(cs_file);
		USES_CONVERSION;
		pcfile = CT2A(cs_file.GetBuffer());
		if (0 == strcmp("", pcfile.data()))
		{
			MessageBox(L"The file path is empty, please check again.", L"Load Info", MB_OK | MB_ICONERROR);
			m_stc_St.SetWindowText(L"Loading failed: empty file path");
		}
		else
		{
			int f_error = -1;
			
			m_edt_xLB.GetWindowTextW(xlb);
			xmin = stof(xlb.GetBuffer());
			m_edt_xUB.GetWindowTextW(xub);
			xmax = stof(xub.GetBuffer());
			m_edt_yStep.GetWindowTextW(ystep);
			ys = stof(ystep.GetBuffer());
			m_edt_zLB.GetWindowTextW(zlb);
			zmin = stof(zlb.GetBuffer());
			m_edt_zUB.GetWindowTextW(zub);
			zmax = stof(zub.GetBuffer());
			m_edt_Width.GetWindowTextW(width);
			w = stoi(width.GetBuffer());
			m_edt_Height.GetWindowTextW(height);
			h = stoi(height.GetBuffer());


			QueryPerformanceCounter(&nst);
			f_error = m_tpc.LoadTyrePC(pcfile,xmin,xmax,ys,zmin,zmax,w,h);
			QueryPerformanceCounter(&nend);
			if (-1 == f_error)
			{
				MessageBox(L"Failed to load point cloud data, please try again!", L"LoadError", MB_OK | MB_ICONERROR);
				m_stc_St.SetWindowText(L"Loading failed: PCL function failed");
			}

			str_len = sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);

			LPCWSTR info_wch = A2W(info_str);
			m_stc_St.SetWindowText(info_wch);
		}
	}

	PointCloud<PointXYZ>::Ptr tmpcloud(::new PointCloud<PointXYZ>),cloud(::new PointCloud<PointXYZ>);//Create point cloud pointer.
	tmpcloud = m_tpc.GetOriginalPC();
	size_t beginID, endID;
	CString cur_val;
	m_edt_PCBeginID.GetWindowTextW(cur_val);
	beginID = stoll(cur_val.GetBuffer());
	m_edt_PCEndID.GetWindowTextW(cur_val);
	endID = stoll(cur_val.GetBuffer());
	cloud->points.insert(cloud->points.begin(), tmpcloud->points.begin() + beginID, tmpcloud->points.begin() + min(endID,tmpcloud->points.size()));

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ClrCloud(cloud, 155, 155, 155);
	viewer->setBackgroundColor(0, 0, 0);
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*cloud, centroid);
	int vp;
	viewer->createViewPort(0,0, centroid(0), centroid(1), vp);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud",vp);
	if (m_bRanSeg)
	{
		vector<ModelCoefficients::Ptr> curCoefs;
		m_tpc.GetReferenceCoefficients(curCoefs);
		for (size_t CoefIt = 0; CoefIt < curCoefs.size(); CoefIt++)
		{
			if (curCoefs[CoefIt]->values.size() == 4)
			{
				viewer->addPlane(*curCoefs[CoefIt]);
			}
			else if (curCoefs[CoefIt]->values.size() == 7)
			{
				viewer->addCylinder(*curCoefs[CoefIt]);
			}
		}
	}
	//viewer->addArrow(cloud->points[0], cloud->points[cloud->points.size() - 1], 255, 0, 0);
	viewer->addLine(cloud->points[0], cloud->points[cloud->points.size() - 1], 155.0, 155.0, 155.0);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	
	while (!viewer->wasStopped())
	{
		viewer->spinOnce();
		pcl_sleep(0.01);
	}
}


void CMFCForTPCDlg::OnBnClickedRadioPins()
{
	// TODO: 在此添加控件通知处理程序代码
	m_RadioID = 1;
}


void CMFCForTPCDlg::OnBnClickedRadioChars()
{
	// TODO: 在此添加控件通知处理程序代码
	m_RadioID = 2;
}
