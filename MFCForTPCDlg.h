
// MFCForTPCDlg.h : ͷ�ļ�
//

#pragma once

#undef min
#undef max

#include "TyrePointCloud.h"
#include "afxcmn.h"
#include "afxwin.h"
#include <atlconv.h>



// CMFCForTPCDlg �Ի���
class CMFCForTPCDlg : public CDialogEx
{
// ����
public:
	CMFCForTPCDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MFCFORTPC_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��


// ʵ��
protected:
	HICON m_hIcon;

	// ���ɵ���Ϣӳ�亯��
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
};
