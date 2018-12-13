
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
}

BEGIN_MESSAGE_MAP(CMFCForTPCDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_Run, &CMFCForTPCDlg::OnBnClickedBtnRun)
	ON_BN_CLICKED(ID_BTN_Exit, &CMFCForTPCDlg::OnBnClickedBtnExit)
	ON_BN_CLICKED(IDC_BTN_SaveData, &CMFCForTPCDlg::OnBnClickedBtnSavedata)
	ON_BN_CLICKED(IDC_BUTTON2, &CMFCForTPCDlg::OnBnClickedButton2)
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
	m_edt_DownSamR.SetWindowTextA("300");
	m_edt_NumThds.SetWindowTextA("2");
	m_edt_DisThrhd.SetWindowTextA("500");
	m_edt_NormDisWt.SetWindowTextA("0.2");
	m_edt_InlR.SetWindowTextA("0.05");
	m_edt_ClTol.SetWindowTextA("300");

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
	m_btn_run.EnableWindow(FALSE);
	m_btn_savedata.EnableWindow(FALSE);
	CFileDialog fileopendlg(TRUE);
	CString filepath;
	if (fileopendlg.DoModal() == IDOK)
	{
		filepath = fileopendlg.GetPathName();
		m_stc_FlPth.SetWindowTextA(filepath);
	}
	else
	{
		m_stc_FlPth.SetWindowTextA("");
	}
	
	// Load point cloud file
	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	QueryPerformanceFrequency(&nfreq);
	PointCloud<PointXYZ>::Ptr cloud(::new PointCloud<PointXYZ>);//Create point cloud pointer.
	CString	cs_file = "";
	string pcfile = "";
	char* info_str = new char[1000];
	int str_len = 0;
	m_stc_FlPth.GetWindowTextA(cs_file);
	pcfile = CT2A(cs_file.GetBuffer());
	if (0 == strcmp("", pcfile.data()))
	{
		MessageBox("The file path is empty, please check again.", "Load Info", MB_OK | MB_ICONERROR);
		m_stc_St.SetWindowTextA("Loading failed: empty file path");
	}
	else
	{
		int f_error = -1;
		
		QueryPerformanceCounter(&nst);
		f_error = m_tpc.LoadTyrePC(pcfile);
		QueryPerformanceCounter(&nend);
		if (-1 == f_error)
		{
			MessageBox("Failed to load point cloud data, please try again!", "LoadError", MB_OK | MB_ICONERROR);
			m_stc_St.SetWindowTextA("Loading failed: PCL function failed");
		}
		
		str_len = sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
		m_stc_St.SetWindowTextA(info_str);
	}
	cloud = m_tpc.GetOriginalPC();

	str_len = sprintf(info_str + str_len, "Starting pin searching, please wait...\n");
	m_stc_St.SetWindowTextA(info_str);
	PointCloud<PointXYZI>::Ptr pins;
	QueryPerformanceCounter(&nst);
	m_tpc.FindPinsBySegmentation(cloud, pins);
	QueryPerformanceCounter(&nend);
	memset(info_str, 0, sizeof(info_str)/sizeof(char));
	char* tmp_str=new char[1000];
	sprintf(tmp_str, "Searching pins costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
	info_str = strcat(info_str, tmp_str);
	m_stc_St.SetWindowTextA(info_str);
	size_t ii = 0;
	while (ii < pins->points.size() && ii < 5)
	{
		memset(tmp_str, 0, sizeof(tmp_str) / sizeof(char));
		sprintf(tmp_str, "%.2f, %.2f, %.2f, %.2f\n",pins->points[ii].x, pins->points[ii].y, pins->points[ii].z, pins->points[ii].intensity);
		info_str = strcat(info_str, tmp_str);
		ii++;
	}
	m_stc_St.SetWindowTextA(info_str);
	m_btn_run.EnableWindow(TRUE);
	m_btn_savedata.EnableWindow(TRUE);

	//Pointers clearation.
	delete[] tmp_str;
	delete[] info_str;
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
	*/
	CDialogEx::OnOK();
}

int CMFCForTPCDlg::SaveXYZToPLYFile(vector<PointCloud<PointXYZ>::Ptr> in_pc, string ex_info)
{
	string savepath, ftype;
	GetPathAndType(savepath, ftype);
	int fe = 0, res = 0;
	vector<PointCloud<PointXYZ>::Ptr>::iterator it = in_pc.begin();
	for (it = in_pc.begin(); it < in_pc.end(); it++)
	{
		fe = pcl::io::savePLYFile(savepath + "_" + to_string(int(it - in_pc.begin())) + "_" + ex_info + ftype, **it);
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
		fe = pcl::io::savePLYFile(savepath + "_" + to_string(int(it - in_pc.begin())) + "_" + ex_info + ftype, **it);
		if (fe < 0)
		{
			res = res + fe;
		}
	}
	return res;
}

void CMFCForTPCDlg::GetPathAndType(string & fpath, string & ftype)
{
	CString cs_file;
	m_stc_FlPth.GetWindowTextA(cs_file);

	string path, fullname, fname, savepath;
	size_t pathid = 0, ftypeid = 0;

	path = cs_file.GetBuffer();
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
	vector<PointCloud<PointXYZI>::Ptr> restClusters;
	m_tpc.GetReferencePlanes(refPlanes);
	m_tpc.GetRestClusters(restClusters);
	SaveXYZToPLYFile(refPlanes, "refPl");
	SaveXYZIToPLYFile(restClusters, "restCl");
}


void CMFCForTPCDlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
	CFileDialog fileopendlg(TRUE);
	CString filepath;
	if (fileopendlg.DoModal() == IDOK)
	{
		filepath = fileopendlg.GetPathName();
	}
	//char* filename;
	//strcpy(filename, filepath);
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
	vector<PinObject> res;
	m_tpc.FindPins(buffer, length, res);
}
