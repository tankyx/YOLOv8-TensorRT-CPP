// DetectorTui — interactive terminal launcher/monitor for detect_object_image.
//
// Pick a config from a directory of config_*.ini files, run the detector as a
// child process, and watch live: metrics from status.json (MetricsWriter), the
// rolling log tail, and the exit status. No CLI typing, no separate windows.
//
// Runs on Linux terminals (Konsole etc.) and on Windows (conhost / Windows
// Terminal / PowerShell). The UI is deliberately ASCII-only: it renders the
// same on every codepage, and works on a plain non-UTF-8 console.
//
// The detector itself is untouched; this is a wrapper that spawns it with the
// config file as its single argument, cwd = the directory containing the
// configs (which is where the models and status.json live).

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#endif

#include "IniParser.h"
#include "TuiCharts.h"
#include "TuiEditor.h"
#include "TuiProcess.h"
#include "TuiTerminal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using tui::ChildProcess;
using tui::Key;
using tui::KeyEvent;
using tui::Size;
using tui::Terminal;

// ---------------------------------------------------------------------------
// Ctrl+C / termination: never tear down the terminal from a signal handler.
// The handler only sets a flag; the main loop notices within one tick and
// takes the normal cleanup path (stop child, restore console, exit).
// ---------------------------------------------------------------------------
static std::atomic<bool> g_quit{false};

static void handleSignal(int) { g_quit.store(true); }

#if defined(_WIN32)
static BOOL WINAPI consoleCtrlHandler(DWORD type) {
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) {
        g_quit.store(true);
        return TRUE; // handled; do not terminate mid-frame
    }
    return FALSE;
}
#endif

// ---------------------------------------------------------------------------
// Small string helpers
// ---------------------------------------------------------------------------
static std::string trunc(const std::string &s, size_t n) {
    if (s.size() <= n) return s;
    if (n <= 3) return s.substr(0, n);
    return s.substr(0, n - 3) + "...";
}

static std::string fmt2(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.2f", v);
    return std::string(b);
}

static std::string fmt1(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.1f", v);
    return std::string(b);
}

static std::string sanitize(const std::string &s) {
    std::string out = s;
    for (char &c : out) {
        unsigned char u = static_cast<unsigned char>(c);
        if (u < 32 || u > 126) c = '.';
    }
    return out;
}

// Command keys arrive with their original case (the config editor needs
// verbatim text); shortcuts compare through this.
static char lowerCh(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------
static std::string readFileTail(const fs::path &p, size_t maxBytes) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    std::streamoff sz = f.tellg();
    if (sz > 0) {
        f.seekg(sz > static_cast<std::streamoff>(maxBytes)
                    ? sz - static_cast<std::streamoff>(maxBytes)
                    : 0);
    } else {
        f.clear();
        f.seekg(0);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<std::string> splitLines(const std::string &s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == '\n') {
            out.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// ---------------------------------------------------------------------------
// Tiny JSON field readers (the status.json schema is flat and fixed: we are
// not going to pull in a JSON dependency for seven fields).
// ---------------------------------------------------------------------------
static std::string jsonObject(const std::string &s, const char *key) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos) return {};
    size_t b = s.find('{', p + k.size());
    if (b == std::string::npos) return {};
    size_t e = s.find('}', b);
    if (e == std::string::npos) return {};
    return s.substr(b, e - b + 1);
}

static bool jsonNumber(const std::string &s, const char *key, double &out) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos) return false;
    p = s.find(':', p + k.size());
    if (p == std::string::npos) return false;
    const char *c = s.c_str() + p + 1;
    char *end = nullptr;
    double v = std::strtod(c, &end);
    if (end == c) return false;
    out = v;
    return true;
}

static bool jsonString(const std::string &s, const char *key, std::string &out) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos) return false;
    size_t c = s.find(':', p + k.size());
    if (c == std::string::npos) return false;
    size_t q1 = s.find('"', c + 1);
    if (q1 == std::string::npos) return false;
    size_t q2 = s.find('"', q1 + 1);
    if (q2 == std::string::npos) return false;
    out = s.substr(q1 + 1, q2 - q1 - 1);
    return true;
}

static bool jsonBool(const std::string &s, const char *key, bool &out) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos) return false;
    p = s.find(':', p + k.size());
    if (p == std::string::npos) return false;
    size_t t = s.find("true", p);
    size_t f = s.find("false", p);
    if (t == std::string::npos && f == std::string::npos) return false;
    out = (f == std::string::npos) || (t != std::string::npos && t < f);
    return true;
}

// Parses the additive "boxes":[[x,y,w,h,conf,label],...] tail of status.json.
// A status file from an older detector simply has no such key -> empty vector.
static std::vector<tui::Box> jsonBoxArray(const std::string &s) {
    std::vector<tui::Box> out;
    size_t p = s.find("\"boxes\"");
    if (p == std::string::npos) return out;
    size_t i = s.find('[', p);
    if (i == std::string::npos) return out;
    ++i; // past the outer '['

    auto isSpace = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    };

    while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || isSpace(s[i]))) ++i; // separators
        if (i >= s.size() || s[i] == ']') break; // end of the box array
        if (s[i] != '[') break;

        const size_t j = s.find(']', i);
        if (j == std::string::npos) break;
        const std::string body = s.substr(i + 1, j - i - 1);
        const char *c = body.c_str();
        char *end = nullptr;
        double v[6] = {0, 0, 0, 0, 0, -1};
        bool ok = true;
        for (int n = 0; n < 6; ++n) {
            v[n] = std::strtod(c, &end);
            if (end == c) {
                ok = false;
                break;
            }
            c = end;
            while (*c == ',' || isSpace(*c)) ++c;
        }
        if (ok) {
            tui::Box b;
            b.x = static_cast<float>(v[0]);
            b.y = static_cast<float>(v[1]);
            b.w = static_cast<float>(v[2]);
            b.h = static_cast<float>(v[3]);
            b.conf = static_cast<float>(v[4]);
            b.label = static_cast<int>(v[5]);
            out.push_back(b);
        }
        i = j + 1;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Config discovery
// ---------------------------------------------------------------------------
struct ConfigEntry {
    std::string file;
    std::string model;
    std::string labels;
    std::string metrics; // MetricsStatus key; empty = metrics disabled
    int fps = 0;
    bool parsed = false;
};

// One parsed status.json sample, shared by the list and monitor screens.
struct LiveState {
    bool statusSeen = false; // status.json exists and is readable
    bool fresh = false;      // written within the last 3s
    bool have = false;       // at least one sample accumulated this run
    bool stale = false;      // no update for over 5s
    double up = 0.0;         // detector uptime, seconds
    double det = 0.0;        // detections per frame
    std::string capAvg = "?", capMin = "?", capMax = "?";
    std::string detAvg = "?", detMin = "?", detMax = "?";
    std::string renAvg = "?", renMin = "?", renMax = "?";
    std::string model;
    std::string prec;
    bool graph = false;
    int boxes = 0;
    float nearest = -1.f;
    bool onTarget = false;
};

static std::vector<ConfigEntry> scanConfigs(const fs::path &dir) {
    std::vector<ConfigEntry> out;
    std::error_code ec;
    fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec);
    if (ec) return out;
    for (const auto &de : it) {
        std::error_code fec;
        if (!de.is_regular_file(fec)) continue;
        std::string name = de.path().filename().string();
        if (name.size() < 8) continue;
        if (name.compare(0, 7, "config_") != 0) continue;
        if (de.path().extension() != ".ini") continue;
        ConfigEntry e;
        e.file = name;
        INIParser ini;
        if (ini.loadFile(de.path().string())) {
            e.model = ini.getString("ModelPath");
            e.labels = ini.getString("Labels");
            e.metrics = ini.getString("MetricsStatus");
            e.fps = ini.getInt("CaptureFPS", 0);
            e.parsed = true;
        }
        out.push_back(std::move(e));
    }
    std::sort(out.begin(), out.end(),
              [](const ConfigEntry &a, const ConfigEntry &b) { return a.file < b.file; });
    return out;
}

// Directory of this executable (so the TUI can be launched from anywhere).
static fs::path executableDir() {
#if defined(_WIN32)
    wchar_t buf[4096];
    DWORD n = GetModuleFileNameW(nullptr, buf, 4096);
    if (n > 0 && n < 4096) return fs::path(buf).parent_path();
    return fs::current_path();
#else
    std::error_code ec;
    fs::path p = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) return p.parent_path();
    return fs::current_path();
#endif
}

// Locate scripts\env.bat by walking up from the config directory. The detector
// runs as a child of this process, so it needs the CUDA/OpenCV DLL directories
// that env.bat sets -- even when the TUI itself was started from a shell that
// did not source it (double-click, fresh terminal, etc.).
static fs::path findEnvBat(const fs::path &start) {
    std::error_code ec;
    fs::path p = fs::absolute(start, ec);
    if (ec) return {};
    for (int depth = 0; depth < 8; ++depth) {
        fs::path candidate = p / "scripts" / "env.bat";
        if (fs::exists(candidate, ec)) return candidate;
        fs::path parent = p.parent_path();
        if (parent.empty() || parent == p) break;
        p = parent;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Frame buffer: fixed-size char grid, painted row by row. Row styles are kept
// separate so the renderer can wrap whole rows in ANSI attributes.
// ---------------------------------------------------------------------------
enum RowStyle {
    STYLE_NORMAL = 0,
    STYLE_TITLE,
    STYLE_SECTION,
    STYLE_SELECTED,
    STYLE_DIM,
    STYLE_WARN
};

struct Canvas {
    int rows = 0;
    int cols = 0;
    std::vector<std::string> line;
    std::vector<unsigned char> style;

    Canvas(int r, int c)
        : rows(r), cols(c), line(r, std::string(static_cast<size_t>(c), ' ')),
          style(r, STYLE_NORMAL) {}

    void put(int row, int col, const std::string &s) {
        if (row < 1 || row > rows || col < 1 || col > cols) return;
        int x = col - 1;
        for (size_t i = 0; i < s.size() && x < cols; ++i, ++x) {
            line[static_cast<size_t>(row - 1)][static_cast<size_t>(x)] = s[i];
        }
    }

    void setStyle(int row, unsigned char st) {
        if (row >= 1 && row <= rows) style[static_cast<size_t>(row - 1)] = st;
    }
};

static void render(Terminal &term, const Canvas &cv) {
    std::string out;
    out.reserve(static_cast<size_t>(cv.rows) * static_cast<size_t>(cv.cols + 16));
    out += "\x1b[H";
    for (int r = 0; r < cv.rows; ++r) {
        switch (cv.style[static_cast<size_t>(r)]) {
            case STYLE_TITLE:    out += "\x1b[7m";  break;
            case STYLE_SECTION:  out += "\x1b[36m"; break;
            case STYLE_SELECTED: out += "\x1b[7m";  break;
            case STYLE_DIM:      out += "\x1b[90m"; break;
            case STYLE_WARN:     out += "\x1b[33m"; break;
            default: break;
        }
        out += cv.line[static_cast<size_t>(r)];
        out += "\x1b[0m";
        if (r + 1 < cv.rows) out += "\r\n";
    }
    term.write(out);
    term.flush();
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void printUsage(const char *argv0) {
    std::printf(
        "detector_tui - interactive launcher for detect_object_image\n"
        "\n"
        "Usage: %s [--dir <config-dir>] [--exe <detector-binary>]\n"
        "\n"
        "  --dir  directory holding config_*.ini, models and status.json\n"
        "         (default: the directory containing this executable)\n"
        "  --exe  detector binary to launch\n"
        "         (default: <dir>/detect_object_image)\n"
        "\n"
        "Keys: up/down or j/k select config, enter run, s stop, r rescan, q quit.\n"
        "      m toggles the live detection monitor while a detector is running.\n",
        argv0);
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    fs::path dir;
    fs::path exe;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (a == "--dir" && i + 1 < argc) {
            dir = argv[++i];
        } else if (a == "--exe" && i + 1 < argc) {
            exe = argv[++i];
        } else {
            std::fprintf(stderr, "detector_tui: unknown argument '%s'\n\n", a.c_str());
            printUsage(argv[0]);
            return 2;
        }
    }

    std::error_code aec;
    if (dir.empty()) dir = executableDir();
    dir = fs::absolute(dir, aec);
#if defined(_WIN32)
    if (exe.empty()) exe = dir / "detect_object_image.exe";
#else
    if (exe.empty()) exe = dir / "detect_object_image";
#endif
    exe = fs::absolute(exe, aec);

    // Apply scripts\env.bat to the detector child so it finds the CUDA/OpenCV
    // DLLs regardless of how the TUI itself was launched.
    fs::path envBat = findEnvBat(dir);

    Terminal term;
    if (!term.init()) {
        std::fprintf(stderr,
                     "detector_tui: not attached to a terminal (a real TTY is required).\n"
                     "              Run it from Konsole / PowerShell, not through a pipe.\n");
        return 1;
    }
    term.write("\x1b[?7l"); // no line wrap: full-width frames may touch the last cell

    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);
#if defined(_WIN32)
    SetConsoleCtrlHandler(consoleCtrlHandler, TRUE);
#endif

    std::vector<ConfigEntry> configs = scanConfigs(dir);
    int sel = 0;
    bool running = false;
    bool everRan = false;
    int lastExit = 0;
    double stopDeadline = -1.0; // steady-clock seconds; kill when reached
    std::string message;
    std::chrono::steady_clock::time_point started{};
    ChildProcess child;

    // ---- screens: config list (with action menu), config editor, monitor ----
    enum class Screen { List, Editor, Monitor };
    Screen screen = Screen::List;
    bool menuOpen = false;
    int menuSel = 0;
    const int kMenuItems = 3; // Edit config / Run detector / Back
    tui::IniEditor editor;
    // Newest box snapshot from status.json; refilled on every launch.
    std::vector<tui::Box> lastBoxes;
    double lastMetricsTs = -1.0;
    float lastFrameW = 0.f;
    float lastFrameH = 0.f;
    std::string runningFile; // config file of the current run

    auto nowSec = []() {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };

    auto launch = [&]() {
        if (configs.empty()) {
            message = "no config_*.ini found in " + dir.string();
            return;
        }
        if (running) return;
        const ConfigEntry &cfg = configs[static_cast<size_t>(sel)];
        std::error_code rec;
        fs::remove(dir / "status.json", rec); // drop stale metrics from a previous run
        std::string err;
        if (!child.start(exe.string(), cfg.file, dir.string(),
                         (dir / "detector_tui.log").string(), err,
                         envBat.string())) {
            message = "launch failed: " + err;
            return;
        }
        running = true;
        everRan = true;
        stopDeadline = -1.0;
        started = std::chrono::steady_clock::now();
        message = "running " + cfg.file;
        runningFile = cfg.file;

        // Fresh run: drop the previous run's box snapshot and jump straight to
        // the live monitor.
        lastBoxes.clear();
        lastMetricsTs = -1.0;
        lastFrameW = lastFrameH = 0.f;
        screen = Screen::Monitor;
    };

    auto openEditor = [&]() {
        if (configs.empty()) return;
        std::string err;
        if (editor.load((dir / configs[static_cast<size_t>(sel)].file).string(), err)) {
            screen = Screen::Editor;
            menuOpen = false;
            message = "editing " + configs[static_cast<size_t>(sel)].file;
        } else {
            message = "editor: " + err;
        }
    };

    // Reads status.json once and refreshes the shared box snapshot and the
    // "metrics stale" message. Latency numbers are still surfaced on the list
    // screen; the monitor deliberately drops the sparkline graphs. Returns an
    // empty state when idle or before the first status file appears.
    auto refreshMetrics = [&]() -> LiveState {
        LiveState live;
        if (!running) return live;

        std::string sj = readFileTail(dir / "status.json", 8192);
        if (sj.empty()) return live;
        live.statusSeen = true;

        double ts = 0;
        const bool fresh = jsonNumber(sj, "ts", ts) &&
                           (std::time(nullptr) - static_cast<time_t>(ts)) <= 3;
        live.fresh = fresh;

        const std::string cap = jsonObject(sj, "capture");
        const std::string det = jsonObject(sj, "detect");
        const std::string ren = jsonObject(sj, "render");
        auto readLatency = [](const std::string &obj, std::string &avg, std::string &mn,
                              std::string &mx) {
            double v = 0;
            if (jsonNumber(obj, "avg", v)) avg = fmt2(v);
            if (jsonNumber(obj, "min", v)) mn = fmt2(v);
            if (jsonNumber(obj, "max", v)) mx = fmt2(v);
        };
        readLatency(cap, live.capAvg, live.capMin, live.capMax);
        readLatency(det, live.detAvg, live.detMin, live.detMax);
        readLatency(ren, live.renAvg, live.renMin, live.renMax);

        double dcount = 0;
        jsonString(sj, "model", live.model);
        jsonString(sj, "precision", live.prec);
        jsonNumber(sj, "detections", dcount);
        jsonNumber(sj, "uptime_s", live.up);
        jsonBool(sj, "graph", live.graph);
        live.det = dcount;

        if (fresh && ts != lastMetricsTs) {
            // New sample: refresh the box snapshot handed to the map.
            lastMetricsTs = ts;
            double fw = 0, fh = 0;
            if (jsonNumber(sj, "fw", fw) && jsonNumber(sj, "fh", fh) && fw > 0 && fh > 0) {
                lastFrameW = static_cast<float>(fw);
                lastFrameH = static_cast<float>(fh);
                lastBoxes = jsonBoxArray(sj);
            } else {
                lastFrameW = lastFrameH = 0.f;
                lastBoxes.clear();
            }
        }

        live.have = (lastMetricsTs >= 0.0);
        live.stale = live.have && (std::time(nullptr) - static_cast<time_t>(ts)) > 5.0;
        if (live.stale) {
            message = "metrics stale - no status.json update for over 5s";
        } else if (message.rfind("metrics stale", 0) == 0) {
            message = "running " + runningFile;
        }

        live.boxes = static_cast<int>(lastBoxes.size());
        if (!lastBoxes.empty() && lastFrameW > 0.f && lastFrameH > 0.f) {
            live.nearest = tui::nearestBoxDistance(lastBoxes, lastFrameW, lastFrameH);
            live.onTarget = tui::crosshairInsideBox(lastBoxes, lastFrameW, lastFrameH);
        }
        return live;
    };

    bool quit = false;
    while (!quit) {
        KeyEvent k = term.readKey(200);

        // ---- keys ---------------------------------------------------------
        const char ch = lowerCh(k.ch); // shortcuts ignore shift/caps

        if (screen == Screen::Editor) {
            switch (editor.handleKey(k, message)) {
                case tui::EditorAction::Back:
                    screen = Screen::List;
                    configs = scanConfigs(dir); // the edited config may have moved
                    if (sel >= static_cast<int>(configs.size())) sel = 0;
                    break;
                case tui::EditorAction::SaveAndRun:
                    screen = Screen::List;
                    launch();
                    break;
                case tui::EditorAction::Quit:
                    quit = true;
                    break;
                case tui::EditorAction::None:
                    break;
            }
        } else if (screen == Screen::Monitor) {
            // Live monitor: keep it minimal, but never hide the stop/quit keys.
            switch (k.key) {
                case Key::Escape:
                    screen = Screen::List;
                    break;
                case Key::Char:
                    switch (ch) {
                        case 'm':
                        case 'l':
                            screen = Screen::List;
                            break;
                        case 's':
                            if (running && stopDeadline < 0.0) {
                                child.requestStop();
                                stopDeadline = nowSec() + 6.0;
                                message = "stop requested (waiting for clean exit)";
                            }
                            break;
                        case 'q':
                            quit = true;
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
        } else if (menuOpen) {
            // Action menu for the selected config.
            if (k.key == Key::Up || (k.key == Key::Char && ch == 'k')) {
                menuSel = (menuSel + kMenuItems - 1) % kMenuItems;
            } else if (k.key == Key::Down || (k.key == Key::Char && ch == 'j')) {
                menuSel = (menuSel + 1) % kMenuItems;
            } else if (k.key == Key::Escape) {
                menuOpen = false;
            } else if (k.key == Key::Char && ch == 'q') {
                quit = true;
            } else if (k.key == Key::Char && (ch == 'e' || ch == 'l')) {
                menuSel = (ch == 'e') ? 0 : 1;
                menuOpen = false;
                if (menuSel == 0) {
                    openEditor();
                } else {
                    launch();
                }
            } else if (k.key == Key::Enter) {
                const int pick = menuSel;
                menuOpen = false;
                if (pick == 0) {
                    openEditor();
                } else if (pick == 1) {
                    launch();
                }
            }
            if (configs.empty()) menuOpen = false;
        } else {
            switch (k.key) {
                case Key::Up:
                    if (!running && !configs.empty()) {
                        sel = (sel + static_cast<int>(configs.size()) - 1) %
                              static_cast<int>(configs.size());
                    }
                    break;
                case Key::Down:
                    if (!running && !configs.empty()) {
                        sel = (sel + 1) % static_cast<int>(configs.size());
                    }
                    break;
                case Key::Enter:
                    if (!running && !configs.empty()) {
                        menuOpen = true;
                        menuSel = 0;
                    }
                    break;
                case Key::Char:
                    switch (ch) {
                        case 'k':
                            if (!running && !configs.empty()) {
                                sel = (sel + static_cast<int>(configs.size()) - 1) %
                                      static_cast<int>(configs.size());
                            }
                            break;
                        case 'j':
                            if (!running && !configs.empty()) {
                                sel = (sel + 1) % static_cast<int>(configs.size());
                            }
                            break;
                        case 'e':
                            if (!running && !configs.empty()) openEditor();
                            break;
                        case 'l':
                            if (!running && !configs.empty()) launch();
                            break;
                        case 'm':
                            if (running) screen = Screen::Monitor;
                            break;
                        case 'r':
                            if (!running) {
                                configs = scanConfigs(dir);
                                if (sel >= static_cast<int>(configs.size())) sel = 0;
                                message = "rescanned: " + std::to_string(configs.size()) +
                                          " config(s)";
                            }
                            break;
                        case 's':
                            if (running && stopDeadline < 0.0) {
                                child.requestStop();
                                stopDeadline = nowSec() + 6.0;
                                message = "stop requested (waiting for clean exit)";
                            }
                            break;
                        case 'q':
                            quit = true;
                            break;
                        default:
                            break;
                    }
                    break;
                default:
                    break;
            }
        }

        // ---- child state --------------------------------------------------
        if (running && !child.running()) {
            running = false;
            stopDeadline = -1.0;
            lastExit = child.exitCode();
            if (screen == Screen::Monitor) screen = Screen::List;
            if (static_cast<unsigned>(lastExit) == 0xC0000135u) {
                message = "detector failed to start: missing DLL (env.bat not applied?)";
            } else {
                message = "detector exited (code " + std::to_string(lastExit) + ")";
            }
        }
        if (running && stopDeadline > 0.0 && nowSec() >= stopDeadline) {
            child.kill();
            running = false;
            stopDeadline = -1.0;
            lastExit = child.exitCode();
            if (screen == Screen::Monitor) screen = Screen::List;
            message = "detector killed (no clean exit within 6s)";
        }
        if (g_quit.load(std::memory_order_relaxed)) quit = true;

        // ---- paint --------------------------------------------------------
        Size sz = term.size();
        if (sz.cols < 76 || sz.rows < 20) {
            Canvas cv(sz.rows, sz.cols);
            std::string m1 = "terminal too small";
            std::string m2 = "need at least 76x20, have " + std::to_string(sz.cols) + "x" +
                             std::to_string(sz.rows);
            cv.put(std::max(1, sz.rows / 2), std::max(1, (sz.cols - static_cast<int>(m1.size())) / 2 + 1), m1);
            cv.put(std::max(1, sz.rows / 2 + 1),
                   std::max(1, (sz.cols - static_cast<int>(m2.size())) / 2 + 1), m2);
            cv.put(sz.rows, 1, " q quit");
            cv.setStyle(sz.rows, STYLE_DIM);
            render(term, cv);
            continue;
        }

        // ---- config editor screen ------------------------------------------
        if (screen == Screen::Editor) {
            Canvas ecv(sz.rows, sz.cols);
            const int ER = sz.rows;
            const int EC = sz.cols;
            const int eMsgRow = ER - 1;
            std::string head =
                " Editing " + editor.path() + (editor.dirty() ? "   (modified)" : "");
            ecv.put(1, 1, trunc(head, static_cast<size_t>(EC)));
            ecv.setStyle(1, STYLE_TITLE);
            ecv.put(2, 1,
                    trunc(" Values are edited in place; comments and key order survive.",
                          static_cast<size_t>(EC)));
            ecv.setStyle(2, STYLE_DIM);

            int bodyRows = eMsgRow - 3;
            if (bodyRows < 1) bodyRows = 1;
            editor.ensureVisible(bodyRows);
            const std::vector<std::pair<int, std::string>> rows = editor.visibleLines(bodyRows);
            for (size_t i = 0; i < rows.size(); ++i) {
                const int r = 3 + static_cast<int>(i);
                if (r >= eMsgRow) break;
                ecv.put(r, 1, trunc(sanitize(rows[i].second), static_cast<size_t>(EC)));
                if (rows[i].first == editor.cursor()) ecv.setStyle(r, STYLE_SELECTED);
            }
            if (!message.empty()) {
                ecv.put(eMsgRow, 1, trunc(sanitize(message), static_cast<size_t>(EC)));
                ecv.setStyle(eMsgRow, STYLE_DIM);
            }
            ecv.put(ER, 1, trunc(" " + editor.footer(), static_cast<size_t>(EC)));
            ecv.setStyle(ER, STYLE_DIM);
            render(term, ecv);
            continue;
        }

        // ---- running monitor screen ----------------------------------------
        // Dedicated full-screen view while the detector runs: a latency column
        // on the left, the detection map to its right, a compact live-numbers
        // line and a short log tail.
        if (screen == Screen::Monitor) {
            LiveState live = refreshMetrics();
            Canvas cv(sz.rows, sz.cols);
            const int R = sz.rows;
            const int C = sz.cols;

            std::string stateStr =
                running ? " [RUNNING pid " + std::to_string(child.pid()) + "]"
                        : " [stopped, last exit " + std::to_string(lastExit) + "]";
            cv.put(1, 1,
                   trunc(" YOLO Detector Monitor" + stateStr + "   " + runningFile +
                             "   dir: " + dir.string(),
                         static_cast<size_t>(C)));
            cv.setStyle(1, STYLE_TITLE);

            // Compact live numbers. No latency graphs here by design; the list
            // screen (press m) still shows the capture/detect/render averages.
            std::string stats = "  detections " + fmt1(live.det) + "/frame   boxes " +
                                std::to_string(live.boxes);
            if (live.boxes > 0) {
                stats += "   nearest " + fmt2(live.nearest);
                if (live.onTarget) stats += "   ON TARGET";
            } else {
                stats += "   nearest --";
            }
            stats += "   model " + (live.model.empty() ? std::string("?") : live.model) +
                     " " + (live.prec.empty() ? std::string("?") : live.prec) +
                     "   graph " + (live.graph ? "yes" : "no") +
                     "   up " + std::to_string(static_cast<int>(live.up)) + "s";
            cv.put(2, 1, trunc(stats, static_cast<size_t>(C)));
            cv.setStyle(2, STYLE_DIM);

            cv.put(3, 1,
                   trunc(" Latency column + target map   capture frame, + = crosshair",
                         static_cast<size_t>(C)));
            cv.setStyle(3, STYLE_SECTION);

            const int msgRow = R - 1;
            const int footerRow = R;
            const int logTail = (R >= 34) ? 2 : 1;
            const int logTop = msgRow - logTail; // rows logTop .. msgRow-1

            // Left latency column plus the map to its right. Character cells are
            // about twice as tall as they are wide, so the map keeps a roughly
            // square capture frame.
            const int borderTop = 4;
            int innerH = (logTop - 1) - borderTop - 1;
            if (innerH < 3) innerH = 3;

            const int panelW = 20; // outer width, incl. border
            const int gap = 3;
            const int maxInnerW = std::max(10, C - panelW - gap - 2);
            int innerW = innerH * 2;
            if (innerW > maxInnerW) innerW = maxInnerW;
            if (innerW < 20) innerW = std::min(20, maxInnerW);

            const int borderBot = borderTop + innerH + 1;
            const int borderW = innerW + 2;

            // Panel + map form one centered group so the latency column stays
            // visually attached to the map on wide terminals.
            const int groupW = panelW + gap + borderW;
            const int panelX = std::max(1, (C - groupW) / 2 + 1);
            const int panelInnerW = panelW - 2;
            const std::string panelRule =
                "+" + std::string(static_cast<size_t>(panelInnerW), '-') + "+";
            cv.put(borderTop, panelX, trunc(panelRule, static_cast<size_t>(C)));
            cv.setStyle(borderTop, STYLE_DIM);
            for (int r = borderTop + 1; r < borderBot; ++r) {
                cv.put(r, panelX, "|");
                cv.put(r, panelX + panelW - 1, "|");
            }
            cv.put(borderBot, panelX, trunc(panelRule, static_cast<size_t>(C)));
            cv.setStyle(borderBot, STYLE_DIM);

            // Panel body: per-stage avg/max, plus min when there is room.
            {
                struct Stage {
                    const char *name;
                    const std::string *avg;
                    const std::string *mn;
                    const std::string *mx;
                };
                const Stage stages[3] = {
                    {"capture", &live.capAvg, &live.capMin, &live.capMax},
                    {"detect", &live.detAvg, &live.detMin, &live.detMax},
                    {"render", &live.renAvg, &live.renMin, &live.renMax},
                };
                const bool withMin = innerH >= 14;
                std::vector<std::string> body;
                body.push_back("latency ms");
                body.push_back("");
                for (const Stage &s : stages) {
                    body.push_back(std::string(" ") + s.name);
                    body.push_back("  avg  " + *s.avg);
                    body.push_back("  max  " + *s.mx);
                    if (withMin) body.push_back("  min  " + *s.mn);
                }
                for (size_t i = 0; i < body.size(); ++i) {
                    const int r = borderTop + 1 + static_cast<int>(i);
                    if (r >= borderBot) break;
                    cv.put(r, panelX + 1,
                           trunc(body[i], static_cast<size_t>(panelInnerW)));
                }
            }

            // Map border, immediately right of the latency panel.
            const int borderX = panelX + panelW + gap;
            const std::string hrule =
                "+" + std::string(static_cast<size_t>(innerW), '-') + "+";
            cv.put(borderTop, borderX, trunc(hrule, static_cast<size_t>(C)));
            cv.setStyle(borderTop, STYLE_DIM);
            for (int r = borderTop + 1; r < borderBot; ++r) {
                cv.put(r, borderX, "|");
                cv.put(r, borderX + borderW - 1, "|");
            }
            cv.put(borderBot, borderX, trunc(hrule, static_cast<size_t>(C)));
            cv.setStyle(borderBot, STYLE_DIM);

            if (!lastBoxes.empty() && lastFrameW > 0.f && lastFrameH > 0.f) {
                std::vector<std::string> map =
                    tui::renderBoxMap(lastBoxes, lastFrameW, lastFrameH, innerW, innerH);
                for (size_t i = 0; i < map.size(); ++i) {
                    cv.put(borderTop + 1 + static_cast<int>(i), borderX + 1, map[i]);
                }
            } else {
                const std::string hint =
                    live.statusSeen ? "no boxes detected"
                                    : "waiting for the first status.json ...";
                const int hy = borderTop + 1 + innerH / 2;
                const int hx = borderX + 1 +
                               std::max(0, (innerW - static_cast<int>(hint.size())) / 2);
                cv.put(hy, hx, hint);
            }

            // Short log tail pinned just above the message line.
            if (logTail > 0) {
                std::string logData = readFileTail(dir / "detector_tui.log", 8192);
                std::vector<std::string> lines = splitLines(logData);
                int from = static_cast<int>(lines.size()) - logTail;
                if (from < 0) from = 0;
                for (int i = from; i < static_cast<int>(lines.size()); ++i) {
                    const int r = logTop + (i - from);
                    if (r < 1 || r >= msgRow) break;
                    cv.put(r, 1, trunc(sanitize(lines[static_cast<size_t>(i)]),
                                       static_cast<size_t>(C)));
                    cv.setStyle(r, STYLE_DIM);
                }
            }

            if (!message.empty()) {
                cv.put(msgRow, 1, trunc(sanitize(message), static_cast<size_t>(C)));
                const bool warn = live.stale ||
                                  message.find("failed") != std::string::npos ||
                                  message.find("killed") != std::string::npos;
                cv.setStyle(msgRow, warn ? STYLE_WARN : STYLE_DIM);
            }

            cv.put(footerRow, 1,
                   trunc(running ? " monitor:  m list   s stop   q quit   Ctrl+C quit"
                                 : " monitor:  m list   q quit",
                         static_cast<size_t>(C)));
            cv.setStyle(footerRow, STYLE_DIM);

            render(term, cv);
            continue;
        }

        Canvas cv(sz.rows, sz.cols);
        const int R = sz.rows;
        const int C = sz.cols;
        const int lw = std::min(36, std::max(22, C / 3)); // left pane width
        const int rx = lw + 3;                            // right pane x (1-based)
        const int rw = C - rx + 1;

        // Title bar.
        std::string stateStr;
        if (running) {
            stateStr = " [RUNNING pid " + std::to_string(child.pid()) + "]";
        } else if (everRan) {
            stateStr = " [stopped, last exit " + std::to_string(lastExit) + "]";
        } else {
            stateStr = " [idle]";
        }
        cv.put(1, 1, trunc(" YOLO TensorRT Detector TUI" + stateStr + "   dir: " + dir.string(),
                           static_cast<size_t>(C)));
        cv.setStyle(1, STYLE_TITLE);

        // Left: config list.
        cv.put(3, 1, "Configs (" + std::to_string(configs.size()) + ")");
        cv.setStyle(3, STYLE_SECTION);

        const int listTop = 4;
        const int maxTopBlock = std::max(1, R - 10); // keeps >= 3 log rows + msg + footer
        const int desiredList = std::max(1, static_cast<int>(configs.size()));
        const int listH = std::min(desiredList, maxTopBlock);

        int startIdx = 0;
        if (static_cast<int>(configs.size()) > listH) {
            startIdx = sel - listH / 2;
            if (startIdx < 0) startIdx = 0;
            if (startIdx > static_cast<int>(configs.size()) - listH) {
                startIdx = static_cast<int>(configs.size()) - listH;
            }
        }
        for (int i = 0; i < listH; ++i) {
            int idx = startIdx + i;
            if (idx >= static_cast<int>(configs.size())) break;
            int row = listTop + i;
            std::string mark = (idx == sel) ? "> " : "  ";
            cv.put(row, 1, trunc(mark + configs[static_cast<size_t>(idx)].file,
                                 static_cast<size_t>(lw)));
            if (idx == sel) cv.setStyle(row, STYLE_SELECTED);
        }
        if (configs.empty()) cv.put(listTop, 1, "  (none found)");

        // Right: selection details + live detector status.
        std::vector<std::string> info;
        if (!configs.empty()) {
            const ConfigEntry &cfg = configs[static_cast<size_t>(sel)];
            info.push_back("Selected: " + cfg.file);
            info.push_back("  model:  " + (cfg.model.empty() ? std::string("?") : cfg.model));
            info.push_back("  labels: " + (cfg.labels.empty() ? std::string("?") : cfg.labels) +
                           "   fps: " + std::to_string(cfg.fps));
        } else {
            info.push_back("No config_*.ini in this directory.");
        }
        info.push_back("  exe:    " + exe.string());
        info.push_back("");

        LiveState live;
        if (running) {
            double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
            info.push_back("Detector: RUNNING  pid " + std::to_string(child.pid()) +
                           "  up " + std::to_string(static_cast<int>(elapsed)) + "s");
            live = refreshMetrics();
            if (!live.statusSeen) {
                info.push_back("  metrics: waiting for the first status.json ...");
            } else {
                info.push_back("  capture " + live.capAvg + " ms   detect " + live.detAvg +
                               " ms   render " + live.renAvg + " ms" +
                               (live.fresh ? "" : "   (stale)"));
                info.push_back("  detections " + fmt1(live.det) + "   model " +
                               (live.model.empty() ? "?" : live.model) + "   " +
                               (live.prec.empty() ? "?" : live.prec) + "   graph " +
                               (live.graph ? "yes" : "no") + "   uptime " +
                               std::to_string(static_cast<int>(live.up)) + "s");
            }
            info.push_back("");
            info.push_back("  m monitor   s stop   q quit");
        } else if (everRan) {
            info.push_back("Detector: not running (last exit code " +
                           std::to_string(lastExit) + ")");
        } else {
            info.push_back("Detector: idle");
            info.push_back("");
            info.push_back("  Press enter to launch the selected config.");
        }

        // The right column and the log pane share rows, so the top section is
        // sized to fit whichever needs more room (config list vs. status
        // block) and the status lines are clipped to stay above the log.
        const int topBlock = std::min(std::max(desiredList, static_cast<int>(info.size())),
                                      maxTopBlock);
        for (size_t i = 0; i < info.size(); ++i) {
            int row = 3 + static_cast<int>(i);
            if (row > 3 + topBlock) break;
            cv.put(row, rx, trunc(sanitize(info[i]), static_cast<size_t>(rw)));
        }

        // Lower pane spans the full width below the top section: the rolling
        // log. The live detection map has its own monitor screen (press m).
        const int paneHeaderRow = 5 + topBlock;
        const int paneTop = paneHeaderRow + 1;
        const int msgRow = R - 1;

        std::string paneHead = "Log (" + (dir / "detector_tui.log").filename().string() + ")";
        if (running) paneHead += "   [m: monitor]";
        cv.put(paneHeaderRow, 1, trunc(paneHead, static_cast<size_t>(C)));
        cv.setStyle(paneHeaderRow, STYLE_SECTION);

        const int logLines = msgRow - paneTop;
        if (logLines > 0) {
            std::string logData = readFileTail(dir / "detector_tui.log", 16384);
            std::vector<std::string> lines = splitLines(logData);
            int from = static_cast<int>(lines.size()) - logLines;
            if (from < 0) from = 0;
            for (int i = from; i < static_cast<int>(lines.size()); ++i) {
                int row = paneTop + (i - from);
                if (row >= msgRow) break;
                cv.put(row, 1, trunc(sanitize(lines[static_cast<size_t>(i)]),
                                     static_cast<size_t>(C)));
            }
        }

        // Action menu overlay: drawn last so it wins over the panes.
        if (menuOpen && !configs.empty()) {
            const int mw = 30;
            const int my = 4;
            const std::string rule =
                "+" + std::string(static_cast<size_t>(mw - 2), '-') + "+";
            cv.put(my, 1, trunc(rule, static_cast<size_t>(C)));
            cv.setStyle(my, STYLE_DIM);
            const char *items[kMenuItems] = {"Edit config", "Run detector", "Back"};
            for (int i = 0; i < kMenuItems; ++i) {
                std::string line = "| ";
                line += (i == menuSel) ? "> " : "  ";
                line += items[i];
                while (line.size() < static_cast<size_t>(mw - 1)) line += ' ';
                line += "|";
                const int r = my + 1 + i;
                cv.put(r, 1, trunc(line, static_cast<size_t>(C)));
                if (i == menuSel) cv.setStyle(r, STYLE_SELECTED);
            }
            cv.put(my + 1 + kMenuItems, 1, trunc(rule, static_cast<size_t>(C)));
            cv.setStyle(my + 1 + kMenuItems, STYLE_DIM);
        }

        // Message + footer.
        if (!message.empty()) {
            cv.put(msgRow, 1, trunc(sanitize(message), static_cast<size_t>(C)));
            bool warn = message.find("failed") != std::string::npos ||
                        message.find("killed") != std::string::npos ||
                        (everRan && !running && lastExit != 0 &&
                         message.find("exited") != std::string::npos);
            cv.setStyle(msgRow, warn ? STYLE_WARN : STYLE_DIM);
        }
        std::string keys;
        if (menuOpen) {
            keys = " menu:  up/down choose   enter confirm   e edit   l run   esc close";
        } else if (running) {
            keys = " running:  m monitor   s stop (clean)   q quit (stops detector)   Ctrl+C quit";
        } else {
            keys = " idle:  up/down or j/k select   enter menu   e edit   l run   r rescan   q quit";
        }
        cv.put(R, 1, trunc(keys, static_cast<size_t>(C)));
        cv.setStyle(R, STYLE_DIM);

        render(term, cv);
    }

    // ---- shutdown ---------------------------------------------------------
    if (child.running()) {
        child.stopAndWait(4000);
    }
    term.write("\x1b[?7h"); // restore line wrap
    term.restore();
    return 0;
}
