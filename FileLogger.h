#pragma once
#include <fstream>
#include <string>
#include <time.h>

using namespace std;

typedef enum LoggerLevel {
	ERROR_t,
	WARN_t,
	INFO_t,
	DEBUG_t,
	TRACE_t
}Level_t;

class FileLogger
{
private:
	ofstream m_of;
	string m_path;
	string m_name;
public:
	FileLogger();
	FileLogger(const string name);
	FileLogger(const string path, const string name);
	~FileLogger();

	void SetPath(const string path);

	void SetName(const string name);

	template<class T>
	void WriteLineToFile(T context, Level_t level = Level_t::DEBUG_t);

	template<class T>
	void WriteToFilfe(T context);
};

