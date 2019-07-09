#pragma once
class IniHelper
{
protected:
	struct INIStruct{
		CString Path;
		CString Name;			 
	};
public:
	IniHelper();
	~IniHelper();

	INIStruct Inipath;

	void Iniwrite(CString sectong, CString key, CString value);
	void Iniwrite(CString sectong, CString *key, CString *value);

	CString Iniread(CString sectong, CString key);
	CString* Iniread(CString sectong, CString *key);

};

