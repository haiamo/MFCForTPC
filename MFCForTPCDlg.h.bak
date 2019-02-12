// MFCForTPCDlg.h : 头文件
//

#pragma once

#undef min
#undef max

#include "stdafx.h"
#include "TyrePointCloud.h"
#include "afxcmn.h"
#include "afxwin.h"
#include "RangImagDlg.h"

#include <pcl\visualization\common\float_image_utils.h>
#include <pcl\io\png_io.h>
#include <pcl\io\io.h>

#include <atlconv.h>

using namespace pcl;


//GPU Device Interfaces:
//int CharToValueDev(thrust::device_vector<string> in_str, pcl::gpu::PtrSz<PointXYZ>& out_pt);

// CMFCForTPCDlg 对话框
class CMFCForTPCDlg : public CDialogEx
{
// 构造
public:
	CMFCForTPCDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MFCFORTPC_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	CEdit m_edt_ClTol;
	CEdit m_edt_DownSamR;
	CEdit m_edt_NormDisWt;
	CEdit m_edt_InlR;
	CEdit m_edt_NumThds;
	CEdit m_edt_DisThrhd;
	afx_msg void OnBnClickedBtnRun();
	CStatic m_stc_FlPth;
	CStatic m_stc_St;
	CButton m_btn_run;
	afx_msg void OnBnClickedBtnExit();

public:
	//User defined methods
	int SaveXYZToPLYFile(vector<PointCloud<PointXYZ>::Ptr> in_pc, string ex_info);
	int SaveXYZIToPLYFile(vector<PointCloud<PointXYZI>::Ptr> in_pc, string ex_info);

	template <typename PointTPtr>
	int SaveCloudToFile(vector<PointTPtr> p_inpc, string ex_info);

	template <typename PointTPtr>
	int SaveCloudToFile(PointTPtr p_inpc, string ex_info);

	void SetSegParameters();

	int LoadPointCloud();

protected:
	//User defined methods
	void GetPathAndType(string &fpath, string &ftype);

private:
	//User defined members
	TyrePointCloud m_tpc;
	RangeImageProperties m_RIProp;

	bool m_bRanSeg = false;
public:
	afx_msg void OnBnClickedBtnSavedata();
	CButton m_btn_savedata;
	afx_msg void OnBnClickedButton2();
//	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedRanimg();
	afx_msg void OnBnClickedShowpc();
};
