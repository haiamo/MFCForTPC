
// MFCForTPC.h : PROJECT_NAME Ӧ�ó������ͷ�ļ�
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"		// ������


// CMFCForTPCApp: 
// �йش����ʵ�֣������ MFCForTPC.cpp
//

class CMFCForTPCApp : public CWinApp
{
public:
	CMFCForTPCApp();

// ��д
public:
	virtual BOOL InitInstance();

// ʵ��

	DECLARE_MESSAGE_MAP()
};

extern CMFCForTPCApp theApp;