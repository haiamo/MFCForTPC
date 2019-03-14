
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

using namespace pcl;


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
	CStatic m_stc_St;
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
};
