#include "stdafx.h"
#include "IniHelper.h"


IniHelper::IniHelper()
{
	Inipath.Path = "";
	Inipath.Name = "log.txt";
}


IniHelper::~IniHelper()
{
}

void IniHelper::Iniwrite(CString  sectong, CString  key, CString value)
{
	CString  lpPath= Inipath.Path+ Inipath.Name;
	WritePrivateProfileString(sectong, key, value, lpPath);
	//delete[] lpPath;
}

void IniHelper::Iniwrite(CString sectong, CString * key, CString * value)
{

	CString  lpPath= Inipath.Path + Inipath.Name;
	int length= key->GetLength();	
	for (int i = 0; i < length; i++)	{
		WritePrivateProfileString(sectong, key[i], value[i], lpPath);
		i++;
	}

}

CString IniHelper::Iniread(CString sectong, CString key)
{
	CString  lpPath = Inipath.Path + Inipath.Name;
	CString def;
	char*re = new char[255];
	GetPrivateProfileString(sectong, key,def,re,8, lpPath);
	return re;
}

CString * IniHelper::Iniread(CString sectong, CString * key)
{
	CString  lpPath = Inipath.Path + Inipath.Name;
	int i = 0;
	CString def;
	int length = key->GetLength();
	char*re = new char[length];	
	CString*restr = new CString[length];
	for (int i = 0; i < length; i++) {
	GetPrivateProfileString(sectong, key[i], def, re, 8, lpPath);
	restr[i] = re;
    }
	return restr;
}
