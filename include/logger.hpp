#ifndef __LOGGER_HPP__
#define __LOGGER_HPP__

#include <cstdarg>
#include <fstream>
#include <string>

class Logger {
public:
    Logger(const std::string& log_file, bool enable);
    ~Logger();
    void log(const char* fmt, ...);

private:
    std::ofstream ofs;
    bool enable;
};

#endif  // __LOGGER_HPP__
