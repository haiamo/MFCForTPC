#include "stdafx.h"
#include "FileLogger.h"


FileLogger::FileLogger()
{
	m_path = "";
	m_name = "log.txt";
	m_of.open(m_name);
}

FileLogger::FileLogger(const string name)
{
	m_path = "";
	m_name = name;
}

FileLogger::FileLogger(const string path, const string name)
{
	m_path = path;
	m_name = name;
}


FileLogger::~FileLogger()
{
	if (m_of.is_open())
	{
		m_of.close();
	}
}

void FileLogger::SetPath(const string path)
{
	m_path = path;
}

void FileLogger::SetName(const string name)
{
	m_name = name;
}

template<class T>
void FileLogger::WriteLineToFile(T context, Level_t level)
{
	time_t curT;
	tm* local;
	char buffer[200] = { 0 };
	curT = time(NULL);
	local = localtime(&curT);
	strftime(buffer, 64, "%Y %m %d %H:%M:%S", local);
	m_of << buffer << endl;
	
	string slvl = "";
	switch (level)
	{
	case Level_t::ERROR_t:
		slvl = "ERROR:";
		break;
	case Level_t::WARN_t:
		slvl = "WARNNING:";
		break;
	case Level_t::INFO_t:
		slvl = "INFORMATION:";
		break;
	case Level_t::DEBUG_t:
		slvl = "DEBUG:";
		break;
	case Level_t::TRACE_t:
		slvl = "TRACE:";
		break;
	}
	m_of << slvl << context << endl;
}

template<class T>
void FileLogger::WriteToFilfe(T context)
{
	m_of << context << " ";
}
