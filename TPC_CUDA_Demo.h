
// TPC_CUDA_Demo.h : main header file for the PROJECT_NAME application
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols


// CTPC_CUDA_DemoApp:
// See TPC_CUDA_Demo.cpp for the implementation of this class
//

class CTPC_CUDA_DemoApp : public CWinApp
{
public:
	CTPC_CUDA_DemoApp();

// Overrides
public:
	virtual BOOL InitInstance();

// Implementation

	DECLARE_MESSAGE_MAP()
};

extern CTPC_CUDA_DemoApp theApp;