
// TPC_CUDA_DemoDlg.cpp : implementation file
//

#include "stdafx.h"
#include "TPC_CUDA_Demo.h"
#include "TPC_CUDA_DemoDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
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


// CTPC_CUDA_DemoDlg dialog



CTPC_CUDA_DemoDlg::CTPC_CUDA_DemoDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_TPC_CUDA_DEMO_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CTPC_CUDA_DemoDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_EDIT_MaxIt, m_edt_MaxIters);
	DDX_Control(pDX, IDC_EDIT_MinInl, m_edt_MinInliers);
	DDX_Control(pDX, IDC_EDIT_ParaSize, m_edt_ParaSize);
	DDX_Control(pDX, IDC_EDIT_LTh, m_edt_LTh);
	DDX_Control(pDX, IDC_EDIT_UTh, m_edt_UTh);
	DDX_Control(pDX, IDC_STC_FilePath, m_stc_FlPth);
	DDX_Control(pDX, IDC_BTN_Run, m_btn_run);
	DDX_Control(pDX, IDC_EDIT_xBeg, m_edt_xBeg);
	DDX_Control(pDX, IDC_EDIT_xEnd, m_edt_xEnd);
	DDX_Control(pDX, IDC_EDIT_PtSize, m_edt_PtSize);
	DDX_Control(pDX, IDC_EDIT_PtSize2, m_edt_PtSize2);
	DDX_Control(pDX, IDC_EDIT_DownSample, m_edt_DSFolder);
	DDX_Control(pDX, IDC_BTN_RunFolder, m_btn_runfolder);
	DDX_Control(pDX, IDC_BTN_Save, m_btn_save);
	DDX_Control(pDX, IDC_BTN_Exit, m_btn_exit);
	DDX_Control(pDX, IDC_EDIT_Status, m_edt_Status);
	DDX_Control(pDX, IDC_BTN_RunAutoCut, m_btn_runac);
	DDX_Control(pDX, IDC_EDT_RANSACMethod, m_edt_ransacMethod);
	DDX_Control(pDX, IDC_STATIC_Height2, m_stc_Height2);
	DDX_Control(pDX, IDC_STATIC_Height3, m_stc_Height3);
	DDX_Control(pDX, IDC_STATIC_PtSize, m_stc_PtSize);
	DDX_Control(pDX, IDC_STATIC_PtSize2, m_stc_PtSize2);
	DDX_Control(pDX, IDC_STATIC_PtSize3, m_stc_PtSize3);
	DDX_Control(pDX, IDC_STATIC_xLB2, m_stc_xLB2);
	DDX_Control(pDX, IDC_STATIC_xUB2, m_stc_xUB2);
	DDX_Control(pDX, IDC_STATIC_yStep2, m_stc_yStep2);
	DDX_Control(pDX, IDC_STATIC_zLB2, m_stc_zLB2);
	DDX_Control(pDX, IDC_STATIC_zUB2, m_stc_zUB2);
	DDX_Control(pDX, IDC_STATIC_zUB3, m_stc_zUB3);
}

BEGIN_MESSAGE_MAP(CTPC_CUDA_DemoDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_Run, &CTPC_CUDA_DemoDlg::OnBnClickedBtnRun)
	ON_BN_CLICKED(IDC_BTN_Exit, &CTPC_CUDA_DemoDlg::OnBnClickedBtnExit)
	ON_BN_CLICKED(IDC_BTN_Save, &CTPC_CUDA_DemoDlg::OnBnClickedBtnSave)
	ON_BN_CLICKED(IDC_BTN_RunFolder, &CTPC_CUDA_DemoDlg::OnBnClickedBtnRunfolder)
	ON_BN_CLICKED(IDC_BTN_RunAutoCut, &CTPC_CUDA_DemoDlg::OnBnClickedBtnRunautocut)
	ON_BN_CLICKED(IDC_BTN_GetDeviceProp, &CTPC_CUDA_DemoDlg::OnBnClickedBtnGetdeviceprop)
	ON_BN_CLICKED(IDC_BTN_Test, &CTPC_CUDA_DemoDlg::OnBnClickedBtnTest)
	ON_EN_CHANGE(IDC_EDT_RANSACMethod, &CTPC_CUDA_DemoDlg::OnEnChangeEdtRansacmethod)
END_MESSAGE_MAP()


// CTPC_CUDA_DemoDlg message handlers

BOOL CTPC_CUDA_DemoDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
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

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	m_edt_xBeg.SetWindowText(_T("0"));
	m_edt_xEnd.SetWindowText(_T("2560"));
	m_edt_PtSize.SetWindowText(_T("0"));
	m_edt_PtSize2.SetWindowText(_T("100000"));
	m_edt_DSFolder.SetWindowText(_T("64"));
	m_edt_ransacMethod.SetWindowText(_T("0"));

	m_edt_MaxIters.SetWindowText(_T("1000"));
	m_edt_MinInliers.SetWindowText(_T("100"));
	m_edt_ParaSize.SetWindowText(_T("4"));
	m_edt_UTh.SetWindowText(_T("5"));
	m_edt_LTh.SetWindowText(_T("5"));

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CTPC_CUDA_DemoDlg::OnSysCommand(UINT nID, LPARAM lParam)
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

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CTPC_CUDA_DemoDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CTPC_CUDA_DemoDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CTPC_CUDA_DemoDlg::OnBnClickedBtnRun()
{
	// TODO: 在此添加控件通知处理程序代码
	EnableAllButtons(FALSE);
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
	float yBeg = 0.0f;

	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	RunFileProp rfProp;
	bool bLoad = true;
	rfProp.Init();
	QueryPerformanceFrequency(&nfreq);
	QueryPerformanceCounter(&nst);
	bLoad = LoadAFile(filepath, yBeg, rfProp);
	QueryPerformanceCounter(&nend);

	if (bLoad)
	{
		QueryPerformanceCounter(&nst);
		RunThroughAFile(filepath,rfProp);
		QueryPerformanceCounter(&nend);

		char* info_str = new char[1000];
		char* tmp_str = new char[1000];
		int str_len = 0;
		USES_CONVERSION;
		string resStr("");
		ReadFilePropIntoStream(rfProp, resStr);
		strcpy(info_str, resStr.c_str());
		LPCWSTR info_wstr = A2W(info_str);
		m_edt_Status.SetWindowTextW(info_wstr);

		delete[] info_str;
		delete[] tmp_str;
		info_str = NULL;
		tmp_str = NULL;
	}
	EnableAllButtons(TRUE);
}


void CTPC_CUDA_DemoDlg::OnBnClickedBtnExit()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnCancel();
}

template<typename PointTPtr>
int CTPC_CUDA_DemoDlg::SaveCloudToFile(vector<PointTPtr> p_inpc, string ex_info, RunFileProp &ioProp)
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
	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	double tmpT = 0.0;
	QueryPerformanceFrequency(&nfreq);
	
	for (vector<PointTPtr>::iterator it = p_inpc.begin(); it < p_inpc.end(); it++)
	{
		QueryPerformanceCounter(&nst);
		fe = pcl::io::savePLYFile(savepath + "_" + to_string(int(it - p_inpc.begin())) + "_" + ex_info + "." + ftype, **it, false);
		QueryPerformanceCounter(&nend);
		tmpT = (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart;
		ioProp.SavedFileName.push_back(to_string(int(it - p_inpc.begin())) + "_" + ex_info);
		ioProp.FileSaveTime.push_back(tmpT);
		ioProp.TotalSaveTime += tmpT;
		if (fe < 0)
		{
			res += fe;
		}
	}
	return res;
}

template<typename PointTPtr>
int CTPC_CUDA_DemoDlg::SaveCloudToFile(PointTPtr p_inpc, string ex_info, RunFileProp &ioProp)
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
	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	double tmpT = 0.0;
	QueryPerformanceFrequency(&nfreq);
	QueryPerformanceCounter(&nst);
	fe = pcl::io::savePLYFile(savepath + "_" + ex_info + "." + ftype, *p_inpc, false);
	QueryPerformanceCounter(&nend);
	tmpT = (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart;
	ioProp.SavedFileName.push_back(ex_info);
	ioProp.FileSaveTime.push_back(tmpT);
	ioProp.TotalSaveTime += tmpT;
	return fe;
}

template<typename DataType>
void CTPC_CUDA_DemoDlg::ConvertMatToText(char *& io_text, const char* begLine, DataType ** inData, int matCols, int matRows, int matNums)
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
				std::sprintf(tmp_str, " %.3lf ", (double)tmpArr[curID]);
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

void CTPC_CUDA_DemoDlg::GetPathAndType(string & fpath, string & ftype)
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

bool CTPC_CUDA_DemoDlg::LoadAFile(CString cs_file, float & yBeg, RunFileProp& io_prop)
{
	// Load point cloud file
	CFileDialog fileopendlg(TRUE);
	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	QueryPerformanceFrequency(&nfreq);
	PointCloud<PointXYZ>::Ptr cloud(::new PointCloud<PointXYZ>);//Create point cloud pointer.
	string pcfile = "", file_type = "";
	char* info_str = new char[1000];
	int str_len = 0;
	if (cs_file == "")
	{
		m_stc_FlPth.GetWindowTextW(cs_file);
	}

	USES_CONVERSION;
	pcfile = CT2A(cs_file.GetBuffer());
	if (0 == strcmp("", pcfile.data()))
	{
		MessageBox(L"The file path is empty, please check again.", L"Load Info", MB_OK | MB_ICONERROR);
		m_edt_Status.SetWindowTextW(L"Loading failed: empty file path");
		return false;
	}
	else
	{
		int f_error = -1;
		CString xbeg, xend;
		float xb, xe, yb = yBeg;
		double tmpT = 0.0;

		m_edt_xBeg.GetWindowTextW(xbeg);
		xb = stof(xbeg.GetBuffer());
		m_edt_xEnd.GetWindowTextW(xend);
		xe = stof(xend.GetBuffer());

		file_type = pcfile.substr(pcfile.length() - 4, 4);
		size_t lastNpos = 0;
		lastNpos = pcfile.find_last_of("\\");
		io_prop.FileName = pcfile.substr(lastNpos + 1, pcfile.length() - lastNpos);

		if (0 == strcmp(file_type.data(), ".dat"))
		{
			string xmlStr = "";
			xmlStr = pcfile.substr(0, pcfile.length() - 4) + ".xml";
			m_tpcProp.SetTPCLoadProp(xmlStr);
		}

		QueryPerformanceCounter(&nst);
		if (0 == strcmp(file_type.data(), ".dat"))
		{
			f_error = m_tpc.LoadTyrePC(pcfile, m_tpcProp, xb, xe, yb);
		}
		else if (0 == strcmp(file_type.data(), ".ply"))
		{
			f_error = m_tpc.LoadTyrePC(pcfile, m_tpcProp);
		}
		
		QueryPerformanceCounter(&nend);
		if (-1 == f_error)
		{
			MessageBox(L"Failed to load point cloud data, please try again!", L"LoadError", MB_OK | MB_ICONERROR);
			m_edt_Status.SetWindowTextW(L"Loading failed: PCL function failed");
		}
		yBeg = yb;
		tmpT = (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0;
		io_prop.LoadTime = tmpT;
		io_prop.TotalPtNum = (m_tpc.GetOriginalPC())->points.size();
		m_tpc.SetTPCProp(m_tpcProp);
		return true;
	}
}

void CTPC_CUDA_DemoDlg::RunThroughAFile(CString cs_file, RunFileProp& io_prop)
{
	/* Load point cloud file
	CFileDialog fileopendlg(TRUE);
	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	QueryPerformanceFrequency(&nfreq);
	PointCloud<PointXYZ>::Ptr cloud(::new PointCloud<PointXYZ>);//Create point cloud pointer.
	string pcfile = "", file_type = "";
	char* info_str = new char[1000];
	int str_len = 0;
	if (cs_file == "")
	{
		m_stc_FlPth.GetWindowTextW(cs_file);
	}
	
	USES_CONVERSION;
	pcfile = CT2A(cs_file.GetBuffer());
	if (0 == strcmp("", pcfile.data()))
	{
		MessageBox(L"The file path is empty, please check again.", L"Load Info", MB_OK | MB_ICONERROR);
		m_edt_Status.SetWindowTextW(L"Loading failed: empty file path");
		m_btn_run.EnableWindow(TRUE);
		return;
	}
	else
	{
		int f_error = -1;
		CString xbeg, xend;
		float xb, xe, yb = 0.0f;

		m_edt_xBeg.GetWindowTextW(xbeg);
		xb = stof(xbeg.GetBuffer());
		m_edt_xEnd.GetWindowTextW(xend);
		xe = stof(xend.GetBuffer());

		file_type = pcfile.substr(pcfile.length() - 4, 4);
		if (0 == strcmp(file_type.data(), ".dat"))
		{
			string xmlStr = "";
			xmlStr = pcfile.substr(0, pcfile.length() - 4) + ".xml";
			m_tpcProp.SetTPCLoadProp(xmlStr);
		}

		QueryPerformanceCounter(&nst);
		f_error = m_tpc.LoadTyrePC(pcfile, m_tpcProp, xb, xe, yb);
		QueryPerformanceCounter(&nend);
		if (-1 == f_error)
		{
			MessageBox(L"Failed to load point cloud data, please try again!", L"LoadError", MB_OK | MB_ICONERROR);
			m_edt_Status.SetWindowTextW(L"Loading failed: PCL function failed");
		}

		str_len = std::sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
	}
	char* tmp_str = new char[1000];
	str_len = std::sprintf(info_str + str_len, "Starting Characteristics searching, please wait...\n");
	LPCWSTR info_wstr = A2W(info_str);
	m_edt_Status.SetWindowTextW(info_wstr);*/
	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	double tmpT = 0.0;
	QueryPerformanceFrequency(&nfreq);
	string pcfile = "", file_type = "";
	char* info_str = new char[1000];
	int str_len = 0;
	USES_CONVERSION;
	char* tmp_str = new char[1000];
	LPCWSTR info_wstr;

	CString maxIt, minInlier, paraSize, UTh, LTh, ransacMethod;
	m_edt_MaxIters.GetWindowTextW(maxIt);
	m_edt_MinInliers.GetWindowTextW(minInlier);
	m_edt_ParaSize.GetWindowTextW(paraSize);
	m_edt_UTh.GetWindowTextW(UTh);
	m_edt_LTh.GetWindowTextW(LTh);
	m_edt_ransacMethod.GetWindowTextW(ransacMethod);
	pcl::PointCloud<PointXYZI>::Ptr pins(::new pcl::PointCloud<PointXYZI>);
	pcl::PointCloud<PointXYZI>::Ptr seghgh(::new pcl::PointCloud<PointXYZI>);
	pcl::PointCloud<PointXYZ>::Ptr chars(::new pcl::PointCloud<PointXYZ>);
	pcl::PointCloud<PointXYZ>::Ptr base(::new pcl::PointCloud<PointXYZ>);
	PointCloud<PointXYZ>::Ptr cloud(::new PointCloud<PointXYZ>);//Create point cloud pointer.
	pcl::PointCloud<PointXYZ>::Ptr inCloud(::new pcl::PointCloud<PointXYZ>);
	cloud = m_tpc.GetOriginalPC();
	pcl::PointCloud<PointXYZ>::iterator listBeg, listEnd;
	CString ptsize_cs;
	size_t ptSizeBeg = 0, ptSizeEnd = 0, totalPCSize = cloud->points.size(), PCPieces = (totalPCSize + PIECEPOINTSIZE - 1) / PIECEPOINTSIZE;
	int dsFolder;
	m_edt_PtSize.GetWindowTextW(ptsize_cs);
	ptSizeBeg = stoll(ptsize_cs.GetBuffer());
	m_edt_PtSize2.GetWindowTextW(ptsize_cs);
	ptSizeEnd = stoll(ptsize_cs.GetBuffer());
	m_edt_DSFolder.GetWindowTextW(ptsize_cs);
	dsFolder = stoi(ptsize_cs.GetBuffer());

	double* paraList = NULL;
	m_tpc.DownSampling(cloud, dsFolder, ptSizeBeg, ptSizeEnd, m_tpcProp);
	cloud = m_tpc.GetDownSample();
	totalPCSize = cloud->points.size();
	PCPieces = (totalPCSize + PIECEPOINTSIZE - 1) / PIECEPOINTSIZE;
	io_prop.DownsamplePtNum = cloud->points.size();
	io_prop.Pieces = PCPieces;
	io_prop.PieceRunTime.reserve(PCPieces);

	std::memset(info_str, 0, sizeof(info_str) / sizeof(char));
	std::memset(tmp_str, 0, sizeof(tmp_str) / sizeof(char));
	if (dsFolder > 1)
	{
		std::sprintf(tmp_str, "The entire Point Cloud is downsampling by %d folders.\r\n", dsFolder);
		info_str = strcat(info_str, tmp_str);
		std::sprintf(tmp_str, "The downsampling Cloud is split into %zd pieces,\r\n whose computing time are shown below:\r\n", PCPieces);
		info_str = strcat(info_str, tmp_str);
	}
	else
	{
		std::sprintf(tmp_str, "The entire Point Cloud is split into %zd pieces,\r\n whose computing time are shown below:\r\n", PCPieces);
		info_str = strcat(info_str, tmp_str);
	}
	info_wstr = A2W(info_str);
	m_edt_Status.SetWindowTextW(info_wstr);
	UpdateData(FALSE);
	UpdateWindow();

	double* ctrPtx, *ctrPty;
	Point3Dw* ctrPts = NULL;

	for (int tt = 0; tt < PCPieces; tt++)
	{
		ptSizeBeg = tt*PIECEPOINTSIZE;
		if (tt < PCPieces - 1)
		{
			ptSizeEnd = ptSizeBeg + PIECEPOINTSIZE;
		}
		else
		{
			ptSizeEnd = ptSizeBeg + totalPCSize%PIECEPOINTSIZE;
		}

		inCloud->points.clear();
		for (size_t ptID = ptSizeBeg; ptID < ptSizeEnd; ptID++)
		{
			//if (ptID%dsFolder == 0)
			//{
				inCloud->points.push_back(cloud->points[ptID]);
			//}
		}

		QueryPerformanceCounter(&nst);
		switch (stoi(ransacMethod.GetBuffer()))
		{
		case 0:
			paraList = new double[stoi(paraSize.GetBuffer())];
			m_tpc.FindCharsBy2DRANSACGPU(inCloud, stoi(maxIt.GetBuffer()), stoi(minInlier.GetBuffer()),
				stoi(paraSize.GetBuffer()), stof(UTh.GetBuffer()), stof(LTh.GetBuffer()), chars, base, paraList);
			/*int maxInt = stoi(maxIt.GetBuffer()), minInt = stoi(minInlier.GetBuffer());
			size_t* idList = new size_t[maxInt*minInt];
			srand(time(NULL));
		
			for (size_t ii = 0; ii < maxInt*minInt; ii++)
			{
				idList[ii] = rand() / RAND_MAX * inCloud->points.size();
			}*/
			break;
		case 1:
			paraList = new double[stoi(paraSize.GetBuffer())];
			ctrPtx = new double[stoi(minInlier.GetBuffer())];
			ctrPty = new double[stoi(minInlier.GetBuffer())];
			m_tpc.FindCharsBy2DRANSACGPUStep(inCloud, stoi(maxIt.GetBuffer()), stoi(minInlier.GetBuffer()),
				stoi(paraSize.GetBuffer()), stof(UTh.GetBuffer()), stof(LTh.GetBuffer()), chars, base,ctrPtx,ctrPty, paraList);
			break;
		case 2:
			paraList = new double[stoi(paraSize.GetBuffer())];
			ctrPtx = new double[stoi(minInlier.GetBuffer()) + 2];
			ctrPty = new double[stoi(minInlier.GetBuffer()) + 2];
			m_tpc.SetClusterTolerance(min(stof(UTh.GetBuffer()), stof(LTh.GetBuffer())));
			m_tpc.FindPinsByNURBSRANSAC(inCloud, stoi(maxIt.GetBuffer()), stoi(minInlier.GetBuffer()), stof(UTh.GetBuffer()),
				stof(LTh.GetBuffer()), pins, base, ctrPtx, ctrPty);
			break;
		case 3:
			paraList = new double[stoi(paraSize.GetBuffer())];
			ctrPtx = new double[stoi(minInlier.GetBuffer())];
			ctrPty = new double[stoi(minInlier.GetBuffer())];
			m_tpc.SetClusterTolerance(min(stof(UTh.GetBuffer()), stof(LTh.GetBuffer())));
			m_tpc.FindSegmentalHeightByNURBSRANSCA(inCloud, stoi(maxIt.GetBuffer()), stoi(minInlier.GetBuffer()), stof(UTh.GetBuffer()),
				stof(LTh.GetBuffer()), seghgh, base, ctrPtx, ctrPty);
			break;
		case 4:
			paraList = new double[stoi(paraSize.GetBuffer())];
			ctrPtx = new double[stoi(minInlier.GetBuffer())];
			ctrPty = new double[stoi(minInlier.GetBuffer())];
			//ctrPts=new Point3Dw[]
			float beg, end, origin, step;
			size_t width, height;
			m_tpcProp.GetAxisProp(&beg, &end, &step, &origin, 'x');
			width = (size_t)(end - beg) / step + 1;
			m_tpcProp.GetAxisProp(&beg, &end, &step, &origin, 'y');
			height = (size_t)(end - beg) / step + 1;
			ctrPts = new Point3Dw[(width + 2)*height];
			const char* info = m_tpc.SplitBaseByNURBSAlongLine(inCloud, stoi(maxIt.GetBuffer()), stoi(minInlier.GetBuffer()), stof(UTh.GetBuffer()),
				stof(LTh.GetBuffer()),(unsigned int)stoi(paraSize.GetBuffer()), pins, base, ctrPts);
			if (strcmp(info, "Success") != 0)
			{
				MessageBox(A2W(info), L"Error", MB_OK | MB_ICONERROR);
				return;
			}
			break;

		}

		QueryPerformanceCounter(&nend);
		tmpT = (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0;
		io_prop.PieceRunTime.push_back(tmpT);
		io_prop.TotalRunTime += tmpT;
		std::memset(tmp_str, 0, sizeof(tmp_str) / sizeof(char));
		std::sprintf(tmp_str, "Part %d costs %.3fs.\r\n", tt + 1, tmpT);
		info_str = strcat(info_str, tmp_str);
		info_wstr = A2W(info_str);
		m_edt_Status.SetWindowTextW(info_wstr);

		UpdateData(FALSE);
		UpdateWindow();
	}
	std::memset(tmp_str, 0, sizeof(tmp_str) / sizeof(char));
	/*QueryPerformanceCounter(&nst);
	//m_tpc.GenerateHypoBaseSurface(paraList, stoi(paraSize.GetBuffer()));
	//m_tpc.GenerateHypoBaseSurface(paraList, stoi(paraSize.GetBuffer()), cloud);

	GridProp inGP;
	float beg, end, step, origin;
	size_t width, height;
	m_tpcProp.GetAxisProp(&beg, &end, &step, &origin, 'x');
	inGP.xbeg = origin;
	inGP.xend = origin + (end - beg)*step;
	inGP.xstep = step*dsFolder;
	width = (size_t)(end - beg) / step + 1;
	m_tpcProp.GetAxisProp(&beg, &end, &step, &origin, 'y');
	inGP.ybeg = origin;
	inGP.yend = origin + (end - beg)*step;
	inGP.ystep = step;
	height = (size_t)(end - beg) / step + 1;
	//m_tpc.GenearteHypoNURBSCurve(inGP, ctrPtx, ctrPty, stoi(minInlier.GetBuffer()));
	//m_tpc.GenerateHypoNURBSCurve(inGP, ctrPts, width + 2, height);
	QueryPerformanceCounter(&nend);
	tmpT = (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0;
	std::memset(tmp_str, 0, sizeof(tmp_str) / sizeof(char));
	std::sprintf(tmp_str, "Hypobase generating costs %.3fs.\r\n", tmpT);
	info_str = strcat(info_str, tmp_str);
	info_wstr = A2W(info_str);
	m_edt_Status.SetWindowTextW(info_wstr);*/


	std::sprintf(tmp_str, "All computing work has been finished.");
	info_str = strcat(info_str, tmp_str);
	info_wstr = A2W(info_str);
	m_edt_Status.SetWindowTextW(info_wstr);
	UpdateData(FALSE);
	UpdateWindow();

	size_t ii = 0;
	pcl::PointCloud<PointXYZ>::Ptr segPC = m_tpc.GetSegPC();
	pcl::PointCloud<PointXYZ>::Ptr restPC = m_tpc.GetRestPC();
	pcl::PointCloud<PointXYZ>::Ptr hypoPC = m_tpc.GetHypoBasePC();
	float beg, end, step, origin;
	float xlb, xub, ylb, yub, zlb, zub;
	m_tpcProp.GetAxisBoundary(&xlb, &xub, 'x');
	m_tpcProp.GetAxisBoundary(&ylb, &yub, 'y');
	m_tpcProp.GetAxisBoundary(&zlb, &zub, 'z');
	for (pcl::PointCloud<PointXYZ>::iterator it = hypoPC->points.begin(); it < hypoPC->points.end();)
	{
		if ((it->x<xlb || it->x>xub) || (it->y<ylb || it->y>yub) || (it->z<zlb || it->z>zub))
		{
			it = hypoPC->points.erase(it);
		}
		else
		{
			it++;
		}
	}
	vector<pcl::PointCloud<PointXYZI>::Ptr> pinsPC;
	pcl::PointCloud<PointXYZI>::Ptr resPins = m_tpc.GetPinsPC();
	m_tpc.GetRestClusters(pinsPC);
	string curName;
	//QueryPerformanceCounter(&nst);
	curName = "BasePC_Total";
	SaveCloudToFile(segPC, curName, io_prop);
	curName = "CharsPC_Total";
	SaveCloudToFile(restPC, curName, io_prop);
	curName = "HypoBasePC_Total";
	SaveCloudToFile(hypoPC, curName, io_prop);
	curName = "PinsClusterPC_Total";
	SaveCloudToFile(pinsPC, curName, io_prop);
	curName = "ResultPinsPC_Total";
	SaveCloudToFile(resPins, curName, io_prop);
	//QueryPerformanceCounter(&nend);
	//tmpT = (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0;
	//io_prop.TotalSaveTime = tmpT;

	//Pointers clearation.
	if (NULL != paraList)
	{
		delete[] paraList;
		paraList = NULL;
	}

	if (NULL != info_str)
	{
		delete[] info_str;
		info_str = NULL;
	}

	if (NULL != tmp_str)
	{
		delete[] tmp_str;
		tmp_str = nullptr;
	}

	if (NULL != ctrPts)
	{
		delete[] ctrPts;
		ctrPts = NULL;
	}

	if (NULL != ctrPty)
	{
		delete[] ctrPty;
		ctrPty = NULL;
	}

	cloud.reset();
}

void CTPC_CUDA_DemoDlg::EnableAllButtons(BOOL bEnable)
{
	m_btn_run.EnableWindow(bEnable);
	m_btn_exit.EnableWindow(bEnable);
	m_btn_runfolder.EnableWindow(bEnable);
	m_btn_save.EnableWindow(bEnable);
	m_btn_runac.EnableWindow(bEnable);
}

void CTPC_CUDA_DemoDlg::ReadFilePropIntoStream(RunFileProp in_prop, string & out_str)
{
	stringstream io_ss;
	io_ss.str("");
	io_ss << "File Name " << in_prop.FileName << "." << endl;
	io_ss << "Total Points " << in_prop.TotalPtNum << "." << endl;
	io_ss << "Loading Time " << in_prop.LoadTime << "s." << endl;
	if (in_prop.DownsamplePtNum > 0)
	{
		io_ss << "Down sampling Points " << in_prop.DownsamplePtNum << "." << endl;
	}	
	io_ss << "Point Pieces " << in_prop.Pieces << "." << endl;
	io_ss << "Running Time " << in_prop.TotalRunTime << "s." << endl;
	io_ss << "Running Pieces Time " << endl;
	for (size_t it = 0; it < in_prop.PieceRunTime.size(); it++)
	{
		if (it < in_prop.PieceRunTime.size() - 1)
		{
			io_ss << in_prop.PieceRunTime[it] << "s, ";
		}
		else
		{
			io_ss << in_prop.PieceRunTime[it] << "s." << endl;
		}
	}
	for (size_t ii = 0; ii < in_prop.SavedFileName.size(); ii++)
	{
		io_ss << in_prop.SavedFileName[ii] << " costs:" << in_prop.FileSaveTime[ii] << "s." << endl;
	}
	io_ss << "Total Saving Time " << in_prop.TotalSaveTime << "s." << endl;
	string tmp = "";
	out_str = "";
	size_t dotPos = 0;
	while (io_ss >> tmp)
	{
		dotPos = tmp.find_last_of(".");
		if (dotPos == tmp.length() - 1)
		{
			out_str = out_str + " " + tmp + "\r\n";
		}
		else
		{
			out_str = out_str + " " + tmp;
		}
	}
	io_ss.str("");
}


void CTPC_CUDA_DemoDlg::OnBnClickedBtnSave()
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
	string pcfile = "", file_type = "";
	char* info_str = new char[1000];
	int str_len = 0;
	m_stc_FlPth.GetWindowTextW(cs_file);
	USES_CONVERSION;
	pcfile = CT2A(cs_file.GetBuffer());
	if (0 == strcmp("", pcfile.data()))
	{
		MessageBox(L"The file path is empty, please check again.", L"Load Info", MB_OK | MB_ICONERROR);
		m_edt_Status.SetWindowTextW(L"Loading failed: empty file path");
		m_btn_run.EnableWindow(TRUE);
		return;
	}
	else
	{
		int f_error = -1;
		CString xbeg, xend;
		float xb, xe, yb = 0.0f;

		m_edt_xBeg.GetWindowTextW(xbeg);
		xb = stof(xbeg.GetBuffer());
		m_edt_xEnd.GetWindowTextW(xend);
		xe = stof(xend.GetBuffer());

		file_type = pcfile.substr(pcfile.length() - 4, 4);
		if (0 == strcmp(file_type.data(), ".dat"))
		{
			MessageBox(L"Please open related .xml file for .dat loading.", L"File needed", MB_OK);
			CString xmlpath;
			string xmlStr = "";
			if (fileopendlg.DoModal() == IDOK)
			{
				xmlpath = fileopendlg.GetPathName();
				xmlStr = CT2A(xmlpath.GetBuffer());
				m_tpcProp.SetTPCLoadProp(xmlStr);
			}
		}

		QueryPerformanceCounter(&nst);
		f_error = m_tpc.LoadTyrePC(pcfile, m_tpcProp, xb, xe, yb);
		QueryPerformanceCounter(&nend);
		if (-1 == f_error)
		{
			MessageBox(L"Failed to load point cloud data, please try again!", L"LoadError", MB_OK | MB_ICONERROR);
			m_edt_Status.SetWindowTextW(L"Loading failed: PCL function failed");
		}

		str_len = std::sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);

		LPCWSTR info_wch = A2W(info_str);
		m_edt_Status.SetWindowTextW(info_wch);
		pcl::PointCloud<PointXYZ>::Ptr oriPC = m_tpc.GetOriginalPC();
		RunFileProp tmpProp;
		QueryPerformanceCounter(&nst);
		if (0 > SaveCloudToFile(oriPC, "OriginPC", tmpProp))
		{
			MessageBox(L"Fail to save point cloud data, please check!", L"Save File Error", MB_OK | MB_ICONERROR);
			m_edt_Status.SetWindowTextW(L"Saving failed: .ply file save failed.");
		}
		QueryPerformanceCounter(&nend);
		str_len = std::sprintf(info_str+str_len, "Saving ply file costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
		info_wch = A2W(info_str);
		m_edt_Status.SetWindowTextW(info_wch);
	}
}


void CTPC_CUDA_DemoDlg::OnBnClickedBtnRunfolder()
{
	// TODO: 在此添加控件通知处理程序代码
	EnableAllButtons(FALSE);
	m_tpc.InitCloudData();
	CFileDialog fileopendlg(TRUE);
	CString filename,filepath;
	if (fileopendlg.DoModal() == IDOK)
	{
		filename = fileopendlg.GetPathName();
		filepath = filename.Mid(filename.ReverseFind('\\') + 1);
		filepath = filename.Left(filename.GetLength() - filepath.GetLength());

		_finddata_t FileInfo;
		USES_CONVERSION;
		string pcpath = CT2A(filepath.GetBuffer());
		string tmpPath = pcpath + "*";
		long Handle = _findfirst(tmpPath.c_str(), &FileInfo);

		if (Handle == -1L)
		{
			m_stc_FlPth.SetWindowText(L"Load folder handle error!");
		}
		else
		{
			string pcfile;
			float yBeg = 0.0f;
			RunFileProp tmpProp;
			vector<RunFileProp> vProp;
			pcl::PointCloud<PointXYZ>::Ptr tmpPC;
			string PCName;
			do {
				tmpProp.Init();
				pcfile = pcpath + FileInfo.name;
				filename = pcfile.c_str();
				m_stc_FlPth.SetWindowTextW(filename);
				if (pcfile.substr(pcfile.length() - 4, 4) == ".dat")
				{
					if (LoadAFile(filename, yBeg, tmpProp))
					{
						//RunThroughAFile(filename,tmpProp);
						tmpPC = m_tpc.GetOriginalPC();
						PCName = "origin";
						SaveCloudToFile(tmpPC, PCName, tmpProp);
						vProp.push_back(tmpProp);
					}
				}
			} while (_findnext(Handle, &FileInfo) == 0);

			char* info_str = new char[1024*1024/4];
			string tmpStr = "", resStr = "";
			for (vector<RunFileProp>::iterator it = vProp.begin(); it < vProp.end(); it++)
			{
				tmpStr = "";
				ReadFilePropIntoStream(*it, tmpStr);
				resStr += tmpStr;
				resStr += "\r\n";
			}
			strcpy(info_str, resStr.c_str());
			LPCWSTR info_wstr = A2W(info_str);
			m_edt_Status.SetWindowTextW(info_wstr);
			delete[] info_str;
			info_str = nullptr;
		}
		_findclose(Handle);
	}
	else
	{
		m_stc_FlPth.SetWindowText(L"Empty file!"); 
	}
	EnableAllButtons(TRUE);
}


void CTPC_CUDA_DemoDlg::OnBnClickedBtnRunautocut()
{
	// TODO: 在此添加控件通知处理程序代码
	EnableAllButtons(FALSE);
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
	float yBeg = 0.0f;

	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	RunFileProp rfProp;
	bool bLoad = true;
	rfProp.Init();
	QueryPerformanceFrequency(&nfreq);
	QueryPerformanceCounter(&nst);
	bLoad = LoadAFile(filepath, yBeg, rfProp);
	QueryPerformanceCounter(&nend);

	if (bLoad)
	{
		CString maxIt, minInlier, paraSize, UTh, LTh, ptsize_cs;
		m_edt_MaxIters.GetWindowTextW(maxIt);
		m_edt_MinInliers.GetWindowTextW(minInlier);
		m_edt_ParaSize.GetWindowTextW(paraSize);
		m_edt_UTh.GetWindowTextW(UTh);
		m_edt_LTh.GetWindowTextW(LTh);
				size_t ptSizeBeg = 0, ptSizeEnd = 0;
		m_edt_PtSize.GetWindowTextW(ptsize_cs);
		ptSizeBeg = stoll(ptsize_cs.GetBuffer());
		m_edt_PtSize2.GetWindowTextW(ptsize_cs);
		ptSizeEnd = stoll(ptsize_cs.GetBuffer());
		m_edt_DSFolder.GetWindowTextW(ptsize_cs);
		int dsFolder = stoi(ptsize_cs.GetBuffer());
		pcl::PointCloud<PointXYZ>::Ptr cloud = m_tpc.GetOriginalPC();
		m_tpc.DownSampling(cloud, dsFolder, ptSizeBeg, ptSizeEnd, m_tpcProp);
		cloud = m_tpc.GetDownSample();
		vector<pcl::PointCloud<PointXYZ>::Ptr> charPCs, basePCs;
		m_tpc.FindCharsWithPieces(cloud, m_tpcProp, stoi(maxIt.GetBuffer()), stoi(minInlier.GetBuffer()),
			stoi(paraSize.GetBuffer()), stof(UTh.GetBuffer()), stof(LTh.GetBuffer()), charPCs, basePCs);

		SaveCloudToFile(charPCs, "char", rfProp);
		SaveCloudToFile(basePCs, "base", rfProp);
	}

	EnableAllButtons(TRUE);
}


void CTPC_CUDA_DemoDlg::OnBnClickedBtnGetdeviceprop()
{
	// TODO: 在此添加控件通知处理程序代码
	cudaError_t cudaErr;
	int count;
	cudaErr = cudaGetDeviceCount(&count);
	if (0 == count)
	{
		MessageBox(L"No variable Devices.", L"DeviceError",MB_OK);
		return;
	}
	cudaDeviceProp myDProp;
	for (int ii = 0; ii < count; ii++)
	{
		cudaErr = cudaGetDeviceProperties(&myDProp, ii);
	}
	
	
}

void CTPC_CUDA_DemoDlg::OnBnClickedBtnTest()
{
	// TODO: 在此添加控件通知处理程序代码
	EnableAllButtons(FALSE);
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
	float yBeg = 0.0f;

	LARGE_INTEGER nfreq, nst, nend;//Timer parameters.
	RunFileProp rfProp;
	bool bLoad = true;
	rfProp.Init();
	QueryPerformanceFrequency(&nfreq);
	QueryPerformanceCounter(&nst);
	bLoad = LoadAFile(filepath, yBeg, rfProp);
	QueryPerformanceCounter(&nend);
}


//void CTPC_CUDA_DemoDlg::OnEnChangeEditPtsize2()
//{
//	// TODO:  如果该控件是 RICHEDIT 控件，它将不
//	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
//	// 函数并调用 CRichEditCtrl().SetEventMask()，
//	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。
//
//	// TODO:  在此添加控件通知处理程序代码
//}


void CTPC_CUDA_DemoDlg::OnEnChangeEdtRansacmethod()
{
	// TODO:  如果该控件是 RICHEDIT 控件，它将不
	// 发送此通知，除非重写 CDialogEx::OnInitDialog()
	// 函数并调用 CRichEditCtrl().SetEventMask()，
	// 同时将 ENM_CHANGE 标志“或”运算到掩码中。

	// TODO:  在此添加控件通知处理程序代码
	CString ransacMethod;
	m_edt_ransacMethod.GetWindowTextW(ransacMethod);
	switch (stoi(ransacMethod.GetBuffer()))
	{
	case 0:
	case 1:
	case 2:
	case 3:
		m_stc_yStep2.SetWindowTextW(L"Para Size");
		m_stc_zLB2.SetWindowTextW(L"Lower Threshold");
		m_stc_zUB2.SetWindowTextW(L"Upper Threshold");
		break;
	case 4:
		m_stc_yStep2.SetWindowTextW(L"Curve Size");
		m_stc_zLB2.SetWindowTextW(L"Curvature Threshold");
		m_stc_zUB2.SetWindowTextW(L"Degree Threshold");
		break;
	}
}
