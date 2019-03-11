
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
	DDX_Control(pDX, IDC_STC_Status, m_stc_St);
	DDX_Control(pDX, IDC_BTN_Run, m_btn_run);
	DDX_Control(pDX, IDC_EDIT_xBeg, m_edt_xBeg);
	DDX_Control(pDX, IDC_EDIT_xEnd, m_edt_xEnd);
	DDX_Control(pDX, IDC_EDIT_PtSize, m_edt_PtSize);
}

BEGIN_MESSAGE_MAP(CTPC_CUDA_DemoDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BTN_Run, &CTPC_CUDA_DemoDlg::OnBnClickedBtnRun)
	ON_BN_CLICKED(IDC_BTN_Exit, &CTPC_CUDA_DemoDlg::OnBnClickedBtnExit)
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
	m_edt_xBeg.SetWindowText(_T("110.0"));
	m_edt_xEnd.SetWindowText(_T("144.275"));

	m_edt_MaxIters.SetWindowText(_T("1000"));
	m_edt_MinInliers.SetWindowText(_T("100"));
	m_edt_ParaSize.SetWindowText(_T("4"));
	m_edt_UTh.SetWindowText(_T("0.2"));
	m_edt_LTh.SetWindowText(_T("0.2"));

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
	m_btn_run.EnableWindow(FALSE);
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
	string pcfile = "", file_type="";
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
		return;
	}
	else
	{
		int f_error = -1;
		CString xbeg,xend;
		float xb,xe;

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
		f_error = m_tpc.LoadTyrePC(pcfile, m_tpcProp, xb, xe);
		QueryPerformanceCounter(&nend);
		if (-1 == f_error)
		{
			MessageBox(L"Failed to load point cloud data, please try again!", L"LoadError", MB_OK | MB_ICONERROR);
			m_stc_St.SetWindowText(L"Loading failed: PCL function failed");
		}

		str_len = std::sprintf(info_str, "Loading text costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);

		/*LPCWSTR info_wch = A2W(info_str);
		m_stc_St.SetWindowText(info_wch);
		pcl::PointCloud<PointXYZ>::Ptr oriPC = m_tpc.GetOriginalPC();
		if (0 > SaveCloudToFile(oriPC, "OriginPC"))
		{
			MessageBox(L"Fail to save point cloud data, please check!", L"Save File Error", MB_OK | MB_ICONERROR);
			m_stc_St.SetWindowTextW(L"Saving failed: .ply file save failed.");
		}*/
	}
	char* tmp_str = new char[1000];
	str_len = std::sprintf(info_str + str_len, "Starting Characteristics searching, please wait...\n");
	LPCWSTR info_wstr = A2W(info_str);
	m_stc_St.SetWindowText(info_wstr);

	CString maxIt, minInlier, paraSize, UTh, LTh;
	m_edt_MaxIters.GetWindowTextW(maxIt);
	m_edt_MinInliers.GetWindowTextW(minInlier);
	m_edt_ParaSize.GetWindowTextW(paraSize);
	m_edt_UTh.GetWindowTextW(UTh);
	m_edt_LTh.GetWindowTextW(LTh);
	pcl::PointCloud<PointXYZ>::Ptr chars(::new pcl::PointCloud<PointXYZ>);
	pcl::PointCloud<PointXYZ>::Ptr base(::new pcl::PointCloud<PointXYZ>);
	pcl::PointCloud<PointXYZ>::Ptr inCloud(::new pcl::PointCloud<PointXYZ>);
	cloud = m_tpc.GetOriginalPC();
	CString ptsize_cs;
	size_t ptSize;

	m_edt_PtSize.GetWindowTextW(ptsize_cs);
	ptSize = stof(ptsize_cs.GetBuffer());
	for (size_t it = 0; it < min(ptSize,cloud->points.size()); it++)
	{
		inCloud->points.push_back(cloud->points[it]);
	}

	QueryPerformanceCounter(&nst);
	m_tpc.FindCharsBy2DRANSACGPU(inCloud, stoi(maxIt.GetBuffer()), stoi(minInlier.GetBuffer()),
		stoi(paraSize.GetBuffer()), stof(UTh.GetBuffer()), stof(LTh.GetBuffer()), chars, base);
	QueryPerformanceCounter(&nend);

	std::memset(info_str, 0, sizeof(info_str) / sizeof(char));
	std::sprintf(tmp_str, "Searching chars costs %.3f seconds.\n", (nend.QuadPart - nst.QuadPart)*1.0 / nfreq.QuadPart*1.0);
	info_str = strcat(info_str, tmp_str);
	info_wstr = A2W(info_str);
	m_stc_St.SetWindowText(info_wstr);

	size_t ii = 0;
	pcl::PointCloud<PointXYZ>::Ptr segPC = m_tpc.GetSegPC();
	pcl::PointCloud<PointXYZ>::Ptr restPC = m_tpc.GetRestPC();
	SaveCloudToFile(segPC, "BasePC");
	SaveCloudToFile(restPC, "CharsPC");

	m_btn_run.EnableWindow(TRUE);

	//Pointers clearation.
	delete[] info_str;
	delete[] tmp_str;
	tmp_str = nullptr;
	info_str = nullptr;
	cloud.reset();
}


void CTPC_CUDA_DemoDlg::OnBnClickedBtnExit()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnCancel();
}

template<typename PointTPtr>
int CTPC_CUDA_DemoDlg::SaveCloudToFile(vector<PointTPtr> p_inpc, string ex_info)
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
int CTPC_CUDA_DemoDlg::SaveCloudToFile(PointTPtr p_inpc, string ex_info)
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
