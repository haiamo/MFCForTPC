#pragma once

#include "afxwin.h"
#include "TyrePointCloud.h"

#include <string>
#include <vector>

#include <pcl\point_types.h>
#include <pcl\point_cloud.h>

#include <pcl\visualization\common\float_image_utils.h>
#include <pcl\visualization\range_image_visualizer.h>
#include <pcl\visualization\pcl_visualizer.h>

using namespace std;

struct RangeImageProperties
{
	vector<float> SenPos;
	float AngRes;
	float MaxAngWid;
	float MaxAngHgt;
	float NoiseLvl;
	float MinRange;
	int BorderSize;

	RangeImageProperties();
};

// RangImagDlg 对话框

class RangImagDlg : public CDialogEx
{
	DECLARE_DYNAMIC(RangImagDlg)

public:
	RangImagDlg(CWnd* pParent = NULL);   // 标准构造函数
	virtual ~RangImagDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_RANGIMAGE };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

	DECLARE_MESSAGE_MAP()
public:
	CEdit m_edt_SenPosX;
	CEdit m_edt_SenPosY;
	CEdit m_edt_SenPosZ;
	CEdit m_edt_NoiseLvl;
	CEdit m_edt_MinRange;
	CEdit m_edt_MaxAngWid;
	CEdit m_edt_MaxAngHgt;
	CEdit m_edt_BorderSize;
	CEdit m_edt_AngRes;
	virtual BOOL OnInitDialog();

	void GetRIProp(RangeImageProperties& out_pro);
	void SetRIProp();
	void SetRIProp(RangeImageProperties in_pro);
	void SetTPC(TyrePointCloud in_tpc);

private:
	RangeImageProperties m_riProp;
	TyrePointCloud m_tpc;
	bool m_SenPosChanged = false;
public:
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedCancel();
	afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);
	virtual BOOL PreTranslateMessage(MSG* pMsg);
};
