
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
	ON_BN_CLICKED(IDC_RADIO_Chars_Seg, &CMFCForTPCDlg::OnBnClickedRadioCharsSeg)
	ON_BN_CLICKED(IDC_RADIO_Chars_LCCP, &CMFCForTPCDlg::OnBnClickedRadioCharsLccp)
	ON_BN_CLICKED(IDC_RADIO_Chars_CPC, &CMFCForTPCDlg::OnBnClickedRadioCharsCpc)
	ON_BN_CLICKED(IDC_BTN_LoadAndSave, &CMFCForTPCDlg::OnBnClickedBtnLoadandsave)
	ON_BN_CLICKED(IDC_RADIO_Chars_BLS, &CMFCForTPCDlg::OnBnClickedRadioCharsBls)
	ON_BN_CLICKED(IDC_BTN_Run_AllData, &CMFCForTPCDlg::OnBnClickedBtnRunAlldata)
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
	m_edt_NumThds.SetWindowText(_T("1000"));
	m_edt_DisThrhd.SetWindowText(_T("0.09"));
	m_edt_NormDisWt.SetWindowText(_T("4"));
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
		
		str_len = std::sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
		
		LPCWSTR info_wch = A2W(info_str);
		m_stc_St.SetWindowText(info_wch);
	}
	PointCloud<PointXYZ>::Ptr totalPC = m_tpc.GetOriginalPC();
	//PointCloud<PointXYZRGB>::Ptr totalRGB = m_tpc.GetRGBPC();
	//PointCloud<PointXYZRGB>::Ptr rgbCloud(::new pcl::PointCloud<PointXYZRGB>);
	CString beginID, endID;
	m_edt_PCBeginID.GetWindowTextW(beginID);
	m_edt_PCEndID.GetWindowTextW(endID);
	cloud->points.insert(cloud->points.begin(), 
		totalPC->points.begin()+stoi(beginID.GetBuffer()), 
		totalPC->points.begin() + min(totalPC->points.size(),(size_t)stoi(endID.GetBuffer())));
	//rgbCloud->points.insert(rgbCloud->points.begin(), 
		//totalRGB->points.begin() + stoi(beginID.GetBuffer()), 
		//totalRGB->points.begin() + min(totalRGB->points.size(),(size_t)stoi(endID.GetBuffer())));
	m_tpc.SetOriginPC(cloud);
	//m_tpc.setOriginRGBPC(rgbCloud);

	SetSegParameters();
	char* tmp_str = new char[1000];
	if (m_RadioID == 1)
	{
		str_len = std::sprintf(info_str + str_len, "Starting pin searching, please wait...\n");
	}
	else
	{
		str_len = std::sprintf(info_str + str_len, "Starting Characteristics searching, please wait...\n");
	}
		
	LPCWSTR info_wstr = A2W(info_str);
	m_stc_St.SetWindowText(info_wstr);
	PointCloud<PointXYZI>::Ptr pins;
	PointCloud<PointXYZ>::Ptr seg_chars;
	PointCloud<PointXYZL>::Ptr lbl_chars;
	QueryPerformanceCounter(&nst);
	switch (m_RadioID)
	{
		case 1:
			m_tpc.FindPinsBySegmentationGPU(cloud, pins);
			break;
		case 2:
			m_tpc.FindCharsBySegmentationGPU(cloud, seg_chars);
			break;
		case 3:
			m_tpc.FindCharsByLCCP(cloud, lbl_chars);
			break;
		case 4:
			m_tpc.FindCharsByCPC(cloud, lbl_chars);
			break;
		case 5:
			CString iters,threshold, inilerRatio, order;
			m_edt_NumThds.GetWindowTextW(iters);
			m_edt_DisThrhd.GetWindowTextW(threshold);
			m_edt_InlR.GetWindowTextW(inilerRatio);
			m_edt_NormDisWt.GetWindowTextW(order);
			//m_tpc.Get2DBaseLineBYRANSAC(cloud, 50, stoi(iters.GetBuffer()), stoi(order.GetBuffer()),stof(threshold.GetBuffer()), (int)cloud->points.size()*stof(inilerRatio.GetBuffer()));
			pcl::PointCloud<PointXYZ>::Ptr chars(::new pcl::PointCloud<PointXYZ>);
			pcl::PointCloud<PointXYZ>::Ptr base(::new pcl::PointCloud<PointXYZ>);
			m_tpc.FindCharsBy2DRANSACGPU(cloud, stoi(iters.GetBuffer()), stoi(inilerRatio.GetBuffer()),
				stoi(order.GetBuffer())+1, stof(threshold.GetBuffer()), stof(threshold.GetBuffer()),chars,base);
			break;
	}
	QueryPerformanceCounter(&nend);
	std::memset(info_str, 0, sizeof(info_str)/sizeof(char));
		
	if (m_RadioID == 1)
	{
		std::sprintf(tmp_str, "Searching pins costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
	}
	else
	{
		std::sprintf(tmp_str, "Searching chars costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
	}
		
	info_str = strcat(info_str, tmp_str);
	info_wstr = A2W(info_str);
	m_stc_St.SetWindowText(info_wstr);
	size_t ii = 0;
	switch (m_RadioID)
	{
		case 1:
			while (ii < pins->points.size() && ii < 5)
			{
				std::memset(tmp_str, 0, sizeof(tmp_str) / sizeof(char));
				std::sprintf(tmp_str, "%.2f, %.2f, %.2f, %.2f\n", pins->points[ii].x, pins->points[ii].y, pins->points[ii].z, pins->points[ii].intensity);
				info_str = strcat(info_str, tmp_str);
				ii++;
			}
			info_wstr = A2W(info_str);
			m_stc_St.SetWindowText(info_wstr);
			break;
		case 2:
			pcl::io::savePLYFile("chars.ply", *seg_chars);
			break;
		case 3:
			pcl::io::savePLYFile("chars_LCCP.ply", *lbl_chars);
			break;
		case 4:
			pcl::io::savePLYFile("chars_CPC.ply", *lbl_chars);
			break;
		case 5:
			pcl::PointCloud<PointXYZ>::Ptr segPC = m_tpc.GetSegPC();
			pcl::PointCloud<PointXYZ>::Ptr restPC = m_tpc.GetRestPC();
			SaveCloudToFile(segPC, "BaseLine");
			SaveCloudToFile(restPC, "RestPC");
			break;
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
	bool bIsBin = false;
	if (ftype == "dat")
	{
		bIsBin = true;
		ftype = "ply";
	}
	for (vector<PointTPtr>::iterator it = p_inpc.begin(); it < p_inpc.end(); it++)
	{

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
	bool bIsBin = false;
	if (ftype == "dat")
	{
		bIsBin = true;
		ftype = "ply";
	}
	fe = pcl::io::savePLYFile(savepath + "_" + ex_info + "." + ftype, *p_inpc, bIsBin);
	return 0;
}

template<typename DataType>
void CMFCForTPCDlg::ConvertMatToText(char *& io_text, const char* begLine, DataType ** inData, int matCols, int matRows, int matNums)
{
	//Input data matrix should be stored in column-major.
	char* tmp_str = (char*)malloc(sizeof(char) * 10000);
	if (strlen(io_text) <= 0)
	{
		std::sprintf(io_text, begLine);
	}
	else
	{
		std::sprintf(tmp_str, begLine);
		io_text = strcat(io_text, tmp_str);
	}
	int curID = 0;
	double* tmpArr;
	for (int ii = 0; ii < matNums; ii++)
	{
		tmpArr = *(inData + ii);
		for (int rowi = 0; rowi < matRows; rowi++)
		{
			for (int coli = 0; coli < matCols; coli++)
			{
				curID = rowi*matCols + coli;
				std::sprintf(tmp_str, " %.3lf ",(double)tmpArr[curID]);
				io_text = strcat(io_text, tmp_str);
				tmp_str[0] = '\0';
			}
			if (matRows > 1)
			{
				std::sprintf(tmp_str, "\n");
				io_text = strcat(io_text, tmp_str);
				tmp_str[0] = '\0';
			}

		}
		if (matNums > 1)
		{
			std::sprintf(tmp_str, "\n");
			io_text = strcat(io_text, tmp_str);
			tmp_str[0] = '\0';
		}
	}
	std::sprintf(tmp_str, "\n");
	io_text = strcat(io_text, tmp_str);
	tmp_str[0] = '\0';
	free(tmp_str);
	tmp_str = NULL;
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

		str_len = std::sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);

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
	CString pcS, iters, hypopts,order;
	m_edt_NormDisWt.GetWindowTextW(iters);
	m_edt_NumThds.GetWindowTextW(pcS);
	m_edt_Width.GetWindowTextW(hypopts);
	m_edt_Height.GetWindowTextW(order);

	double *xvals, *yvals, **paras,*modelErr, **dists,*hypox,*hypoy,*As=NULL,**Qs=NULL,**taus=NULL,**Rs=NULL;
	size_t pcsize = stoi(pcS.GetBuffer()), its = stoi(iters.GetBuffer()), hypos = stoi(hypopts.GetBuffer()), paraSize = stoi(order.GetBuffer());
	xvals = (double*)malloc(sizeof(double) * pcsize);
	memset(xvals, 0.0, sizeof(double)*pcsize);
	yvals = (double*)malloc(sizeof(double) * pcsize);
	memset(yvals, 0.0, sizeof(double)*pcsize);
	paras = (double**)malloc(sizeof(double*) * its);
	modelErr = (double*)malloc(sizeof(double) * its);
	memset(modelErr, 0.0, sizeof(double)*its);
	hypox = (double*)malloc(sizeof(double) * its * hypos);
	hypoy = (double*)malloc(sizeof(double) * its * hypos);
	As = (double*)malloc(sizeof(double)* its * hypos * paraSize);
	memset(As, 0.0, sizeof(double)*its * hypos * paraSize);
	Qs = (double**)malloc(sizeof(double*)* its);
	taus = (double**)malloc(sizeof(double*)* its);
	Rs = (double**)malloc(sizeof(double*)*its);
	dists = (double**)malloc(sizeof(double*)*its);

	for (int i = 0; i < its; i++)
	{
		Qs[i] = (double*)malloc(sizeof(double) * hypos * hypos);
		memset(Qs[i], 0.0, sizeof(double)* hypos * hypos);
		taus[i] = (double*)malloc(sizeof(double) * paraSize);
		memset(taus[i], 0.0, sizeof(double)* paraSize);
		Rs[i] = (double*)malloc(sizeof(double)*hypos*paraSize);
		memset(Rs[i], 0.0, sizeof(double)* hypos * paraSize);
		paras[i] = (double*)malloc(sizeof(double)*paraSize);
		memset(paras[i], 0.0, sizeof(double)*paraSize);
		dists[i] = (double*)malloc(sizeof(double)*pcsize);
		memset(dists[0], 0.0,sizeof(double)*pcsize);
	}

	srand(time(NULL));
	double tmpRand = 0.0;
	for (int ii = 0; ii < pcsize; ii++)
	{
		tmpRand = rand()*1.0 / RAND_MAX*0.1;
		xvals[ii] = ii / 10.0;
		yvals[ii] = ii * ii*0.02 + ii*(0.3 + tmpRand) + 1.0 + tmpRand;
	}

	if (cudaError_t::cudaSuccess != RANSACOnGPU(xvals, yvals, pcsize, its, hypos, paraSize, hypox, hypoy, As, Qs, taus, Rs, paras,modelErr,dists))
	{
		return;
	}
	/*else
	{
		cudaError_t cudaErr;
		cublasHandle_t handle;
		cublasStatus_t stat;
		int num = its, rows = hypos, cols = paraSize, ltau = cols;//1,3,3,3

		double **Aarray, **Tauarray = taus;
		Aarray = (double**)malloc(sizeof(double*));
		Aarray[0] = (double*)malloc(sizeof(double)*rows*cols);
		srand(time(0));
		for (int ii = 0; ii < rows; ii++)
		{
			for (int jj = 0; jj < cols; jj++)
			{
				Aarray[0][ii*rows + jj] = rand() % 100;
			}
		}

		//Create host pointer array to device matrix storage
		double **d_Aarray, **d_Tauarray, **h_d_Aarray, **h_d_Tauarray;
		double **d_outA, **h_d_outA;
		h_d_Aarray = (double**)malloc(num * sizeof(double*));
		h_d_Tauarray = (double**)malloc(num * sizeof(double*));
		h_d_outA = (double**)malloc(num * sizeof(double*));

		for (int i = 0; i < num; i++)
		{
			//cudaErr = cudaMalloc((void**)&h_d_Aarray[i], rows*cols * sizeof(double));
			//cudaErr = cudaMalloc((void**)&h_d_Tauarray[i], ltau * sizeof(double));
			stat = cublasAlloc(rows*cols, sizeof(double), (void**)&h_d_Aarray[i]);
			stat = cublasAlloc(ltau, sizeof(double), (void**)&h_d_Tauarray[i]);
			stat = cublasAlloc(rows*cols, sizeof(double), (void**)&h_d_outA[i]);
		}

		//Copy the host array of device pointers to the device
		//cudaErr = cudaMalloc((void**)&d_Aarray, num * sizeof(double*));
		//cudaErr = cudaMalloc((void**)&d_Tauarray, num * sizeof(double*));
		stat = cublasAlloc(num, sizeof(double*), (void**)&d_Aarray);
		stat = cublasAlloc(num, sizeof(double*), (void**)&d_Tauarray);
		stat = cublasAlloc(num, sizeof(double*), (void**)&d_outA);

		stat = cublasCreate(&handle);
		
		//Set input matrices onto device
		for (int i = 0; i < num; i++)
		{
			stat = cublasSetMatrix(rows, cols, sizeof(double), Aarray[i], rows, h_d_Aarray[i], rows);
			stat = cublasSetVector(ltau, sizeof(double), Tauarray[i], 1, h_d_Tauarray[i], 1);
			stat = cublasSetMatrix(rows, cols, sizeof(double), Aarray[i], rows, h_d_outA[i], rows);
			//cudaErr = cudaMemcpy(h_d_Aarray[i], Aarray[i], rows*cols * sizeof(double), cudaMemcpyHostToDevice);
			//cudaErr = cudaMemcpy(h_d_Tauarray[i], Tauarray[i], ltau * sizeof(double), cudaMemcpyHostToDevice);
			//cudaErr = cudaMemcpy(d_Aarray[i], Aarray[i], rows*cols * sizeof(double), cudaMemcpyHostToDevice);
			//cudaErr = cudaMemcpy(d_Tauarray[i], Tauarray[i], ltau * sizeof(double), cudaMemcpyHostToDevice);
		}
		stat = cublasSetVector(num, sizeof(double*), h_d_Tauarray,1, d_Tauarray,1 );
		stat = cublasSetVector(num, sizeof(double*), h_d_Aarray, 1, d_Aarray, 1);
		stat = cublasSetVector(num, sizeof(double*), h_d_outA, 1, d_outA, 1);
		//cudaErr = cudaMemcpy(d_Aarray, h_d_Aarray, num *sizeof(double*), cudaMemcpyHostToDevice);
		cudaErr = cudaThreadSynchronize();
		int *info, lda = rows;
		//cudaErr = cudaMalloc((void**)&info, sizeof(int)*num);
		info = (int*)malloc(sizeof(int)*num);
		stat = cublasDgeqrfBatched(handle, rows, cols, d_Aarray, lda, d_Tauarray, info, num);
		//stat = cublasDmatinvBatched(handle, rows, &d_Aarray[0], lda, d_outA, lda, info, num);
		cudaErr = cudaThreadSynchronize();
		//Retrieve result matrix from device
		cudaErr = cudaMemcpy(h_d_Aarray, d_Aarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
		cudaErr = cudaMemcpy(h_d_Tauarray, d_Tauarray, num * sizeof(double*), cudaMemcpyDeviceToHost);
		for (int i = 0; i < num; i++)
		{
			stat = cublasGetMatrix(rows, cols, sizeof(double), h_d_Aarray[i], rows, Aarray[i], rows);
			stat = cublasGetVector(ltau, sizeof(double), h_d_Tauarray[i], 1, Tauarray[i], 1);
			//cudaErr = cudaMemcpy(Aarray[i], h_d_Aarray[i], rows*cols * sizeof(double), cudaMemcpyDeviceToHost);
			//cudaErr = cudaMemcpy(Tauarray[i], h_d_Tauarray[i], ltau * sizeof(double), cudaMemcpyDeviceToHost);
			//cudaErr = cudaMemcpy(Aarray[i], d_Aarray[i], rows*cols * sizeof(double), cudaMemcpyDeviceToHost);
			//cudaErr = cudaMemcpy(Tauarray[i], d_Tauarray[i], ltau * sizeof(double), cudaMemcpyDeviceToHost);
			//cudaErr = cudaMemcpy(h_dd_Aarray[i], Aarray[i], rows*cols * sizeof(double), cudaMemcpyHostToDevice);
			//cudaErr = cudaMemcpy(h_dd_Tauarray[i], Tauarray[i], ltau * sizeof(double), cudaMemcpyHostToDevice);
		}
		//cublasGetVector(num, sizeof(double*), d_Tauarray, 1, Tauarray, 1);
		//cudaErr = cudaMemcpy(dd_Aarray, h_dd_Aarray, num * sizeof(double*), cudaMemcpyHostToDevice);
		//cudaErr = cudaMemcpy(dd_Tauarray, h_dd_Tauarray, num * sizeof(double*), cudaMemcpyHostToDevice);

		free(Aarray[0]);
		free(Aarray);

		for (int i = 0; i < num; i++)
		{
			cudaFree(h_d_Aarray[i]);
			cudaFree(h_d_Tauarray[i]);
			cudaFree(h_d_outA[i]);
		}
		free(h_d_Aarray);
		free(h_d_Tauarray);
		free(h_d_outA);

		cublasFree(d_Aarray);
		cublasFree(d_Tauarray);
		cublasFree(d_outA);

		cublasDestroy(handle);
		free(info);
	}*/
	int str_len = 0;
	char* info_str = (char*)malloc(sizeof(char) * 10000000);
	char* tmp_str = (char*)malloc(sizeof(char) * 10000);
	double* tmp_arr;
	int curID = 0;
	/*str_len = std::sprintf(info_str, "Hypo_x:\n");
	for (int ii = 0; ii < its*hypos;ii++)
	{
		std::sprintf(tmp_str, " %.1f ", hypox[ii]);
		if (ii % (hypos) == hypos - 1)
		{
			std::sprintf(tmp_str, "%.1f\n", hypox[ii]);
		}
		info_str = strcat(info_str, tmp_str);
		tmp_str[0] = '\0';
	}
	std::sprintf(tmp_str, "\n");
	info_str = strcat(info_str, tmp_str);
	tmp_str[0] = '\0';

	str_len = std::sprintf(info_str, "Hypo_y:\n");
	for (int ii = 0; ii < its*hypos; ii++)
	{
		std::sprintf(tmp_str, " %.1f ", hypoy[ii]);
		if (ii % (hypos) == hypos - 1)
		{
			std::sprintf(tmp_str, "%.1f\n", hypoy[ii]);
		}
		info_str = strcat(info_str, tmp_str);
		tmp_str[0] = '\0';
	}
	std::sprintf(tmp_str, "\n");
	info_str = strcat(info_str, tmp_str);
	tmp_str[0] = '\0';

	
	std::sprintf(tmp_str, "A^T s:\n");
	info_str = strcat(info_str, tmp_str);
	for (int ii = 0; ii < its; ii++)
	{
		for (int coli = 0; coli < paraSize; coli++)
		{
			for (int rowi = 0; rowi < hypos; rowi++)
			{
				curID = ii*hypos*paraSize + coli*hypos + rowi;
				std::sprintf(tmp_str, " %.3f ", As[curID]);
				info_str = strcat(info_str, tmp_str);
				tmp_str[0] = '\0';
			}
			std::sprintf(tmp_str, "\n");
			info_str = strcat(info_str, tmp_str);
			tmp_str[0] = '\0';
		}
		std::sprintf(tmp_str, "\n");
		info_str = strcat(info_str, tmp_str);
		tmp_str[0] = '\0';
	}
	tmp_str[0] = '\0';

	std::sprintf(tmp_str, "Q^T s:\n");
	info_str = strcat(info_str, tmp_str);
	for (int jj = 0; jj < its; jj++)
	{
		tmp_arr = Qs[jj];
		for (int coli = 0; coli < hypos; coli++)
		{
			for (int rowi = 0; rowi < hypos; rowi++)
			{
				curID = coli*hypos + rowi;
				std::sprintf(tmp_str, " %.3f ", tmp_arr[curID]);
				info_str = strcat(info_str, tmp_str);
				tmp_str[0] = '\0';
			}
			std::sprintf(tmp_str , "\n");
			info_str = strcat(info_str, tmp_str);
			tmp_str[0] = '\0';
		}
		std::sprintf(tmp_str, "\n\n");
		info_str = strcat(info_str, tmp_str);
		tmp_str[0] = '\0';
	}

	tmp_str[0] = '\0';
	std::sprintf(tmp_str, "taus:\n");
	info_str = strcat(info_str, tmp_str);
	for (int jj = 0; jj < its; jj++)
	{
		tmp_arr = taus[jj];
		for (int ii = 0; ii < paraSize; ii++)
		{
			std::sprintf(tmp_str, " %.3f ", tmp_arr[ii]);
			info_str = strcat(info_str, tmp_str);
			tmp_str[0] = '\0';
		}
		std::sprintf(tmp_str, "\n");
		info_str = strcat(info_str, tmp_str);
		tmp_str[0] = '\0';
	}
	std::sprintf(tmp_str, "\n\n");
	info_str = strcat(info_str, tmp_str);
	tmp_str[0] = '\0';

	std::sprintf(tmp_str, "R^T s:\n");
	info_str = strcat(info_str, tmp_str);
	for (int jj = 0; jj < its; jj++)
	{
		tmp_arr = Rs[jj];
		for (int coli = 0; coli < paraSize; coli++)
		{
			for (int rowi = 0; rowi < hypos; rowi++)
			{
				curID = coli*hypos + rowi;
				std::sprintf(tmp_str, " %.3f ", tmp_arr[curID]);
				info_str = strcat(info_str, tmp_str);
				tmp_str[0] = '\0';
			}
			std::sprintf(tmp_str, "\n");
			info_str = strcat(info_str, tmp_str);
			tmp_str[0] = '\0'; 
		}
		std::sprintf(tmp_str, "\n");
		info_str = strcat(info_str, tmp_str);
		tmp_str[0] = '\0';
	}
	std::sprintf(tmp_str, "\n\n");
	info_str = strcat(info_str, tmp_str);*/



	info_str[0] = '\0';
	ConvertMatToText(info_str, "xVals:\n", &xvals, pcsize);
	ConvertMatToText(info_str, "yVals:\n", &yvals, pcsize);
	ConvertMatToText(info_str, "paras:\n", paras, paraSize,1,its);
	ConvertMatToText(info_str, "ModelErrs:\n", &modelErr, its);
	ConvertMatToText(info_str, "Distance:\n", dists, pcsize, 1, its);

	/*std::sprintf(info_str, "xVals:\n");
	for (int ii = 0; ii < pcsize; ii++)
	{
		std::sprintf(tmp_str, " %.1f ", xvals[ii]);
		info_str = strcat(info_str, tmp_str);
		tmp_str[0] = '\0';
	}
	std::sprintf(tmp_str, "\n");
	info_str = strcat(info_str, tmp_str);
	tmp_str[0] = '\0';

	std::sprintf(tmp_str, "yVals:\n");
	info_str = strcat(info_str, tmp_str);
	for (int ii = 0; ii < pcsize; ii++)
	{
		std::sprintf(tmp_str, " %.1f ", yvals[ii]);
		info_str = strcat(info_str, tmp_str);
		tmp_str[0] = '\0';
	}
	std::sprintf(tmp_str, "\n");
	info_str = strcat(info_str, tmp_str);
	tmp_str[0] = '\0';

	std::sprintf(tmp_str, "paras:\n");
	info_str = strcat(info_str, tmp_str);
	for (int jj = 0; jj < its; jj++)
	{
		tmp_arr = paras[jj];
		for (int ii = 0; ii < paraSize; ii++)
		{
			std::sprintf(tmp_str, " %.3f ", tmp_arr[ii]);
			info_str = strcat(info_str, tmp_str);
			tmp_str[0] = '\0';
		}
		std::sprintf(tmp_str, "\n");
		info_str = strcat(info_str, tmp_str);
		tmp_str[0] = '\0';
	}
	std::sprintf(tmp_str, "\n");
	info_str = strcat(info_str, tmp_str);
	tmp_str[0] = '\0';

	std::sprintf(tmp_str, "ModelErrs:\n");
	info_str = strcat(info_str, tmp_str);
	for (int jj = 0; jj < its; jj++)
	{
		std::sprintf(tmp_str, " %.3f ", modelErr[jj]);
		info_str = strcat(info_str, tmp_str);
		tmp_str[0] = '\0';
	}
	std::sprintf(tmp_str, "\n");
	info_str = strcat(info_str, tmp_str);
	tmp_str[0] = '\0';

	std::sprintf(tmp_str, "Distances:\n");
	info_str = strcat(info_str, tmp_str);
	for (int jj = 0; jj < its; jj++)
	{
		tmp_arr = dists[jj];
		for (int coli = 0; coli < pcsize; coli++)
		{
			std::sprintf(tmp_str, " %.3f ", tmp_arr[coli]);
			info_str = strcat(info_str, tmp_str);
			tmp_str[0] = '\0';
		}
		std::sprintf(tmp_str, "\n");
		info_str = strcat(info_str, tmp_str);
		tmp_str[0] = '\0';
	}*/

	USES_CONVERSION;
	LPCWSTR info_wch = A2W(info_str);
	m_stc_St.SetWindowText(info_wch);

	//memset(dists, -1.0, sizeof(double)*pcsize);
	/*paras[0] = 1.0;
	paras[1] = 1.0;
	paras[2] = 1.0;
	paras[3] = 1.0;
	double tmpRand = 0.0, tmpRand1 = 0.0;
	pcl::PointCloud<PointXYZ>::Ptr cloud(::new PointCloud<PointXYZ>);
	vector<int> pcIDs;
	vector<double> xVec, yVec;
	srand((unsigned int)time(0));
	PointXYZ tmpPt;
	for (int ii = 0; ii < pcsize; ii++)
	{
		
		tmpRand = (double)randRange(15,10);
		//xvals[ii] = tmpRand+(double)randRange(0.5,-0.5);
		//yvals[ii] = tmpRand*tmpRand+(double)randRange(1.5,-1.5);
		xVec.push_back(tmpRand);// +(double)randRange(0.05, -0.05));
		yVec.push_back(tmpRand*tmpRand*tmpRand*tmpRand*3.5 + 20.0*tmpRand*tmpRand*tmpRand + 15.0*tmpRand*tmpRand + 2.5*tmpRand + 10.0 +(double)randRange(0.05, -0.05));
		dists[ii] = -1.0;
		cloud->points.push_back(PointXYZ(xVec[ii], 0.0, yVec[ii]));
		pcIDs.push_back(ii);
	}

	for (int ii = 0; ii < 100000; ii++)
	{
		cloud->points.push_back(PointXYZ((double)randRange(50, -10), 0.0, (double)randRange(3500000, 4000)));
	}
	pcl::io::savePLYFile("TestData.ply", *cloud);
	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.

	m_tpc.SetOriginPC(cloud);
	vector<double> outPara;
	//paras=(double*)malloc(sizeof(double)*pcsize);
	m_tpc.PolyFit2D(pcIDs, 4, outPara);
	TRACE("The fitted function is y=%fx^4+%fx^3+%fx^2+%fx+%f.\n", outPara[4], outPara[3], outPara[2], outPara[1], outPara[0]);*/

	/*QueryPerformanceFrequency(&nfreq);
	QueryPerformanceCounter(&nst);
	GetPCIntercept(xvals, yvals, pcsize, paras,4, dists);
	QueryPerformanceCounter(&nend);
	TRACE("GPU series computing costs %f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);

	QueryPerformanceCounter(&nst);
	GetPCInterceptAsynch(xvals, yvals, pcsize, paras, 4, dists);
	QueryPerformanceCounter(&nend);
	TRACE("GPU asynchoronous computing costs %f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);*/
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
	if (NULL != paras)
	{
		for (int i = 0; i < its; i++)
		{
			if (NULL != paras[i])
			{
				free(paras[i]);
			}			
		}
		free(paras);
		paras = NULL;
	}

	if (NULL != modelErr)
	{
		free(modelErr);
		modelErr = NULL;
	}

	if (NULL != dists)
	{
		for (int i = 0; i < its; i++)
		{
			if (NULL != dists[i])
			{
				free(dists[i]);
			}
		}
		free(dists);
		dists = NULL;
	}
	if (NULL != hypox)
	{
		free(hypox);
		hypox = NULL;
	}
	if (NULL != hypoy)
	{
		free(hypoy);
		hypoy = NULL;
	}

	if (NULL != As)
	{
		free(As);
		As = NULL;
	}

	if (NULL != Qs)
	{
		for (int i = 0; i < its; i++)
		{
			if (NULL != Qs[i])
			{
				free(Qs[i]);
			}
		}
		free(Qs);
		Qs = NULL;
	}

	if (NULL != Rs)
	{
		for (int i = 0; i < its; i++)
		{
			if (NULL != Rs[i])
			{
				free(Rs[i]);
			}
		}
		free(Rs);
		Rs=NULL;
	}
	
	if (NULL != taus)
	{
		for (int i = 0; i < its; i++)
		{
			if (NULL != taus[i])
			{
				free(taus[i]);
			}
		}
		free(taus);
		taus = NULL;
	}

	if (NULL != info_str)
	{
		free(info_str);
		info_str = NULL;
	}
	
	if (NULL != tmp_str)
	{
		free(tmp_str);
		tmp_str = NULL;
	}


	/*pcl::PointCloud<PointXYZ>::Ptr Cld_ptr(::new pcl::PointCloud<PointXYZ>);
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
	}*/

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

			str_len = std::sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);

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

void CMFCForTPCDlg::OnBnClickedRadioCharsSeg()
{
	// TODO: 在此添加控件通知处理程序代码
	m_RadioID = 2;
}


void CMFCForTPCDlg::OnBnClickedRadioCharsLccp()
{
	// TODO: 在此添加控件通知处理程序代码
	m_RadioID = 3;
}


void CMFCForTPCDlg::OnBnClickedRadioCharsCpc()
{
	// TODO: 在此添加控件通知处理程序代码
	m_RadioID = 4;
}


void CMFCForTPCDlg::OnBnClickedBtnLoadandsave()
{
	// TODO: 在此添加控件通知处理程序代码
	
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
	else if(0 != strcmp(m_preFilePath.data(), pcfile.data()))
	{
		m_tpc.InitCloudData();
		m_preFilePath = pcfile;
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

		str_len = std::sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);

		LPCWSTR info_wch = A2W(info_str);
		m_stc_St.SetWindowText(info_wch);
	}
	str_len = std::sprintf(info_str + str_len, "Start save point cloud into ply file.\n");
	LPCWSTR info_wch = A2W(info_str);
	m_stc_St.SetWindowText(info_wch);
	QueryPerformanceCounter(&nst);
	PointCloud<PointXYZ>::Ptr totalPC = m_tpc.GetOriginalPC();
	CString beginID, endID;
	m_edt_PCBeginID.GetWindowTextW(beginID);
	m_edt_PCEndID.GetWindowTextW(endID);
	size_t begPT = 0, endPT = 0;
	begPT = max(0, stoi(beginID.GetBuffer()));
	endPT = max((size_t)begPT + 10000, min(totalPC->points.size(), (size_t)stoi(endID.GetBuffer())));
	cloud->points.insert(cloud->points.begin(),
		totalPC->points.begin() + begPT,
		totalPC->points.begin() + endPT);
	SaveCloudToFile(cloud, "origin");
	QueryPerformanceCounter(&nend);

	str_len = std::sprintf(info_str, "Save file costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
	info_wch = A2W(info_str);
	m_stc_St.SetWindowText(info_wch);
}


void CMFCForTPCDlg::OnBnClickedRadioCharsBls()
{
	// TODO: 在此添加控件通知处理程序代码
	m_RadioID = 5;
}


void CMFCForTPCDlg::OnBnClickedBtnRunAlldata()
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

		str_len = std::sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);

		LPCWSTR info_wch = A2W(info_str);
		m_stc_St.SetWindowText(info_wch);
	}
	PointCloud<PointXYZ>::Ptr totalPC = m_tpc.GetOriginalPC();
	size_t begID, endID;
	CString begCS,endCS;
	m_edt_PCBeginID.GetWindowTextW(begCS);
	m_edt_PCEndID.GetWindowTextW(endCS);
	begID = stoll(begCS.GetBuffer());
	size_t psize=stoll(endCS.GetBuffer())-stoll(begCS.GetBuffer())+1, parts = totalPC->points.size() / psize + 1;
	char* tmp_str = new char[1000], *file_name=new char[30];
	LPCWSTR info_wstr;
	SetSegParameters();
	CString iters, threshold, inilerRatio, order;
	m_edt_NumThds.GetWindowTextW(iters);
	m_edt_DisThrhd.GetWindowTextW(threshold);
	m_edt_InlR.GetWindowTextW(inilerRatio);
	m_edt_NormDisWt.GetWindowTextW(order);
	int ITs = stoi(iters.GetBuffer()), Ord = stoi(order.GetBuffer());
	double Thres = stod(threshold.GetBuffer()), InR = stod(inilerRatio.GetBuffer());
	pcl::PointCloud<PointXYZ>::Ptr segPC = m_tpc.GetSegPC();
	pcl::PointCloud<PointXYZ>::Ptr restPC = m_tpc.GetRestPC();
	for (size_t partID = 0; partID < parts; partID++)
	{
		//begID = partID * psize;
		endID = begID + psize - 1;
		m_tpc.InitCloudData();
		cloud->points.clear();
		cloud->points.insert(cloud->points.begin(),
			totalPC->points.begin() + begID,
			totalPC->points.begin() + min(totalPC->points.size(), endID));
		m_tpc.SetOriginPC(cloud);
		if (partID<3 || (partID>=3 && partID < 10 && partID>12))
		{
			switch (partID % 10)
			{
			case 0:
				str_len = std::sprintf(info_str + str_len, "On %d-st Characteristics searching, please wait...\n", partID + 1);
				break;
			case 1:
				str_len = std::sprintf(info_str + str_len, "On %d-nd Characteristics searching, please wait...\n", partID + 1);
				break;
			case 2:
				str_len = std::sprintf(info_str + str_len, "On %d-rd Characteristics searching, please wait...\n", partID + 1);
				break;
			default:
				str_len = std::sprintf(info_str + str_len, "On %d-th Characteristics searching, please wait...\n", partID + 1);
				break;
			}
		}
		else
		{
			str_len = std::sprintf(info_str + str_len, "On %d-th Characteristics searching, please wait...\n", partID + 1);
		}

		info_wstr = A2W(info_str);
		m_stc_St.SetWindowText(info_wstr);
		QueryPerformanceCounter(&nst);
		//m_tpc.Get2DBaseLineBYRANSAC(cloud, 50, ITs, Ord, Thres, (int)cloud->points.size()*InR);
		QueryPerformanceCounter(&nend);
		//std::memset(info_str, 0, sizeof(info_str) / sizeof(char));
		std::sprintf(tmp_str, "Current searching costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);

		info_str = strcat(info_str, tmp_str);
		info_wstr = A2W(info_str);
		m_stc_St.SetWindowText(info_wstr);

		segPC = m_tpc.GetSegPC();
		restPC = m_tpc.GetRestPC();
		std::sprintf(file_name, "BaseLine%d", partID);
		SaveCloudToFile(segPC, file_name);
		std::memset(file_name, 0, sizeof(file_name) / sizeof(char));
		std::sprintf(file_name, "RestPC%d", partID);
		SaveCloudToFile(restPC, file_name);

		std::memset(info_str, 0, sizeof(info_str) / sizeof(char));
		std::memset(tmp_str, 0, sizeof(tmp_str) / sizeof(char));
		str_len = 0;
		begID = endID+1;
	}

	m_btn_run.EnableWindow(TRUE);
	m_btn_savedata.EnableWindow(TRUE);

	//Pointers clearation.

	delete[] info_str;
	delete[] tmp_str;
	delete[] file_name;
	tmp_str = nullptr;
	info_str = nullptr;
	file_name = nullptr;
	cloud.reset();
}
