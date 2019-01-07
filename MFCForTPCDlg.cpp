
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
	m_edt_DownSamR.SetWindowTextW(_T("300"));
	m_edt_NumThds.SetWindowTextW(_T("2"));
	m_edt_DisThrhd.SetWindowTextW(_T("500"));
	m_edt_NormDisWt.SetWindowTextW(_T("0.2"));
	m_edt_InlR.SetWindowTextW(_T("0.05"));
	m_edt_ClTol.SetWindowTextW(_T("300"));

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
		m_stc_FlPth.SetWindowTextW(filepath);
	}
	else
	{
		m_stc_FlPth.SetWindowTextW(_T(""));
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
	pcfile = CT2A(cs_file.GetBuffer());
	if (0 == strcmp("", pcfile.data()))
	{
		MessageBox(_T("The file path is empty, please check again."), _T("Load Info"), MB_OK | MB_ICONERROR);
		m_stc_St.SetWindowTextW(_T("Loading failed: empty file path"));
	}
	else
	{
		int f_error = -1;
		
		QueryPerformanceCounter(&nst);
		f_error = m_tpc.LoadTyrePC(pcfile);
		QueryPerformanceCounter(&nend);
		if (-1 == f_error)
		{
			MessageBox(_T("Failed to load point cloud data, please try again!"), _T("LoadError"), MB_OK | MB_ICONERROR);
			m_stc_St.SetWindowTextW(_T("Loading failed: PCL function failed"));
		}
		
		str_len = sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
		m_stc_St.SetWindowTextW((LPCTSTR)(LPTSTR)info_str);
	}
	cloud = m_tpc.GetOriginalPC();

	SetSegParameters();
	str_len = sprintf(info_str + str_len, "Starting pin searching, please wait...\n");
	m_stc_St.SetWindowTextW((LPCTSTR)(LPTSTR)info_str);
	PointCloud<PointXYZI>::Ptr pins;
	QueryPerformanceCounter(&nst);
	m_tpc.FindPinsBySegmentation(cloud, pins);
	QueryPerformanceCounter(&nend);
	memset(info_str, 0, sizeof(info_str)/sizeof(char));
	char* tmp_str=new char[1000];
	sprintf(tmp_str, "Searching pins costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
	info_str = strcat(info_str, tmp_str);
	m_stc_St.SetWindowTextW((LPCTSTR)(LPTSTR)info_str);
	size_t ii = 0;
	while (ii < pins->points.size() && ii < 5)
	{
		memset(tmp_str, 0, sizeof(tmp_str) / sizeof(char));
		sprintf(tmp_str, "%.2f, %.2f, %.2f, %.2f\n",pins->points[ii].x, pins->points[ii].y, pins->points[ii].z, pins->points[ii].intensity);
		info_str = strcat(info_str, tmp_str);
		ii++;
	}
	m_stc_St.SetWindowTextW((LPCTSTR)(LPTSTR)info_str);
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
		fe = pcl::io::savePLYFile(savepath + "_" + to_string(int(it - p_inpc.begin())) + "_" + ex_info + "." + ftype, **it);
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
	m_tpc.SetDownSampleRaius(stof(cur_val.GetBuffer()));

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
	vector<PointCloud<PointXYZI>::Ptr> restClusters;
	m_tpc.GetReferencePlanes(refPlanes);
	m_tpc.GetRestClusters(restClusters);
	SaveCloudToFile(refPlanes, "refPl");
	SaveCloudToFile(restClusters, "restCl");
	//SaveXYZToPLYFile(refPlanes, "refPl");
	//SaveXYZIToPLYFile(restClusters, "restCl");
}


void CMFCForTPCDlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
	/*CFileDialog fileopendlg(TRUE);
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

	string cur_str, in_str = buffer, tmpstr;
	bool b_pcData = false;
	int pos = 0;
	PointXYZ cur_pt;
	PointXYZRGB cur_rgb;
	float x, y, z;
	int r, g, b;
	int line_pos = 0;
	const char* t1 = " ";
	const char* t2 = "\n";
	string::iterator str_it = in_str.begin();
	PointCloud<PointXYZ>::Ptr cloud=m_tpc.GetOriginalPC();
	PointCloud<PointXYZRGB>::Ptr cloud_rgb=m_tpc.GetRGBPC();

	for(pos=0;pos<length;pos++)//while (pos<length && str_it<in_str.end())
	{
		tmpstr = *str_it;
		if (!b_pcData)
		{
			
			if (tmpstr.compare(" ")==0|| tmpstr.compare("\n")==0)
			{
				if (cur_str != "")
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
				cur_str = cur_str + tmpstr;
			}
		}
		else
		{
			if (tmpstr.compare(" ")==0 || tmpstr.compare("\n")==0)
			{
				if (cur_str != "")
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
					if (tmpstr.compare("\n")==0)
					{
						line_pos = 0;
						cur_pt.x = x;
						cur_pt.y = y;
						cur_pt.z = z;
						cloud->points.push_back(cur_pt);

						cur_rgb.x = x;
						cur_rgb.y = y;
						cur_rgb.z = z;
						cur_rgb.r = r;
						cur_rgb.g = g;
						cur_rgb.b = b;
						cloud_rgb->points.push_back(cur_rgb);
					}
					else
					{
						line_pos++;
					}
				}
			}
			else
			{
				cur_str = cur_str + tmpstr;
			}
		}
		//pos++;
		str_it++;
	}
	vector<PinObject> res;
	SetSegParameters();
	m_tpc.FindPins(buffer, length, res);*/
	//m_tpc.FindPinsMT(buffer, length, res);

	/*CString cur_val;
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
	//runtest(a, b, cg, cursize);//调用显卡加速
	char* gpustr = new char[100];
	sprintf(gpustr, "GPU running time: %fms\n", ((double)(clock() - now1)) / CLOCKS_PER_SEC * 1000);//输出GPU处理时间

	long now2 = clock();//存储图像处理开始时间  
	int dif = 0;
	for (int i = 0; i < cursize; i++)
	{
		c[i] = (a[i] + b[i]);
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
}
