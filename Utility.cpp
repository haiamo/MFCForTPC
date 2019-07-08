#include "stdafx.h"
#include "Utility.h"

int FreeCPtrs(int cnt, ...)
{
	int SucCnt = 0;
	va_list vaList;
	va_start(vaList, cnt);
	void** tmpPtr;
	for (int ii = 0; ii < cnt; ii++)
	{
		tmpPtr = va_arg(vaList, void**);
		if (NULL != *tmpPtr)
		{
			free(*tmpPtr);
			*tmpPtr = NULL;
			SucCnt++;
		}
	}
	return SucCnt;
}

int FreeCplusplusPtrsArr(int cnt, ...)
{
	int SucCnt = 0;
	va_list vaList;
	va_start(vaList, cnt);
	void** tmpPtr;
	for (int ii = 0; ii < cnt; ii++)
	{
		tmpPtr = va_arg(vaList, void**);
		if (NULL != *tmpPtr)
		{
			delete[] *tmpPtr;
			*tmpPtr = NULL;
			SucCnt++;
		}
	}
	return SucCnt;
}
