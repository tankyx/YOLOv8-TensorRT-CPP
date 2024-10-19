#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class INIParser {
private:
    std::unordered_map<std::string, std::string> data;

    // Trim whitespace from beginning and end of string
    static std::string trim(const std::string &str) {
        size_t first = str.find_first_not_of(" \t");
        if (std::string::npos == first) {
            return str;
        }
        size_t last = str.find_last_not_of(" \t");
        return str.substr(first, (last - first + 1));
    }

    // Split a string into a vector of strings based on a delimiter
    static std::vector<std::string> split(const std::string &s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(trim(token));
        }
        return tokens;
    }

public:
    bool loadFile(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == ';' || line[0] == '#') {
                continue; // Skip empty lines and comments
            }

            size_t delimiterPos = line.find('=');
            if (delimiterPos != std::string::npos) {
                std::string key = trim(line.substr(0, delimiterPos));
                std::string value = trim(line.substr(delimiterPos + 1));
                data[key] = value;
            }
        }

        return true;
    }

    std::string getString(const std::string &key, const std::string &defaultValue = "") const {
        auto it = data.find(key);
        return (it != data.end()) ? it->second : defaultValue;
    }

    int getInt(const std::string &key, int defaultValue = 0) const {
        auto it = data.find(key);
        if (it != data.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    bool getBool(const std::string &key, bool defaultValue = false) const {
        auto it = data.find(key);
        if (it != data.end()) {
            std::string value = it->second;
            std::transform(value.begin(), value.end(), value.begin(), ::tolower);
            return (value == "true" || value == "1" || value == "yes" || value == "on");
        }
        return defaultValue;
    }

    float getFloat(const std::string &key, float defaultValue = 0.0f) const {
        auto it = data.find(key);
        if (it != data.end()) {
            try {
                return std::stof(it->second);
            } catch (...) {
                return defaultValue;
            }
        }
        return defaultValue;
    }

    // New method to get an array of strings
    std::vector<std::string> getStringArray(const std::string &key, const std::string &delimiter = ",") const {
        auto it = data.find(key);
        if (it != data.end()) {
            return split(it->second, delimiter[0]);
        }
        return std::vector<std::string>();
    }

    // New method to get an array of strings as a single const string
    std::string getStringArrayAsString(const std::string &key, const std::string &delimiter = ",") const {
        auto it = data.find(key);
        return (it != data.end()) ? it->second : "";
    }
};