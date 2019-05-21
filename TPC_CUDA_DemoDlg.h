
// TPC_CUDA_DemoDlg.h : header file
//

#pragma once
#undef min
#undef max

#include "stdafx.h"
#include "TyrePointCloud.h"
#include "TPCProperty.h"
#include "afxcmn.h"
#include "afxwin.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"
#include "cuda.h"
#include "device_functions.h"
#include "cublas.h"
#include "cublas_v2.h"

#include <pcl\io\png_io.h>
#include <pcl\io\io.h>

#include <atlconv.h>

#define PIECEPOINTSIZE 1000000

using namespace pcl;

struct RunFileProp
{
	string FileName;
	double LoadTime;
	double TotalRunTime;
	double SaveTime;
	vector<double> PieceRunTime;
	size_t TotalPtNum;
	size_t DownsamplePtNum;
	size_t Pieces;

	void Init()
	{
		FileName = "";
		LoadTime = 0.0;
		TotalRunTime = 0.0;
		SaveTime = 0.0;
		PieceRunTime.clear();
		TotalPtNum = 0;
		DownsamplePtNum = 0;
		Pieces = 0;
	}
};


// CTPC_CUDA_DemoDlg dialog
class CTPC_CUDA_DemoDlg : public CDialogEx
{
// Construction
public:
	CTPC_CUDA_DemoDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_TPC_CUDA_DEMO_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtnRun();
	afx_msg void OnBnClickedBtnExit();
//	CEdit m_edt_xLB;
//	CEdit m_edt_xUB;
//	CEdit m_edt_yStep;
//	CEdit m_edt_zLB;
//	CEdit m_edt_zUB;
//	CEdit m_edt_Width;
//	CEdit m_edt_Height;
	CEdit m_edt_MaxIters;
	CEdit m_edt_MinInliers;
	CEdit m_edt_ParaSize;
	CEdit m_edt_LTh;
	CEdit m_edt_UTh;
	CStatic m_stc_FlPth;
//	CStatic m_stc_St;
	CButton m_btn_run;

public:
	//User defined methods
	template <typename PointTPtr>
	int SaveCloudToFile(vector<PointTPtr> p_inpc, string ex_info);

	template <typename PointTPtr>
	int SaveCloudToFile(PointTPtr p_inpc, string ex_info);

protected:
	//User defined methods
	void GetPathAndType(string &fpath, string &ftype);

	template<typename DataType>
	void ConvertMatToText(char*&io_text, const char* begLine, DataType** inData, int matCols, int matRows = 1, int matNums = 1);

	bool LoadAFile(CString cs_file, float& yBeg, RunFileProp& io_prop);

	void RunThroughAFile(CString cs_file, RunFileProp& io_prop);

	void EnableAllButtons(BOOL bEnable);

	void ReadFilePropIntoStream(RunFileProp in_prop, string & out_str);

private:
	//User defined members
	TyrePointCloud m_tpc;
	TPCProperty m_tpcProp;
	string m_preFilePath = "";
public:
	CEdit m_edt_xBeg;
	CEdit m_edt_xEnd;
	CEdit m_edt_PtSize;
	afx_msg void OnBnClickedBtnSave();
	CEdit m_edt_PtSize2;
	CEdit m_edt_DSFolder;
	afx_msg void OnBnClickedBtnRunfolder();
	CButton m_btn_runfolder;
	CButton m_btn_save;
	CButton m_btn_exit;
	CEdit m_edt_Status;
	CButton m_btn_runac;
	afx_msg void OnBnClickedBtnRunautocut();
	afx_msg void OnBnClickedBtnGetdeviceprop();
	CEdit m_edt_ransacMethod;
	afx_msg void OnBnClickedBtnTest();
};
