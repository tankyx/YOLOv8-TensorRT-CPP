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
    ++i;
    while (i < s.size() && s[i] != ']') {
        if (s[i] == '[') {
            size_t j = s.find(']', i);
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
                while (*c == ',' || *c == ' ') ++c;
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
        } else if (s[i] == ',') {
            ++i;
        } else {
            break;
        }
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
        "Keys: up/down or j/k select config, enter run, s stop, r rescan, q quit.\n",
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
    enum class Screen { List, Editor };
    Screen screen = Screen::List;
    bool menuOpen = false;
    int menuSel = 0;
    const int kMenuItems = 3; // Edit config / Run detector / Back
    tui::IniEditor editor;
    // Latency history (one sample per status.json timestamp change) and the
    // newest box snapshot; refilled on every launch.
    std::vector<double> capHist;
    std::vector<double> detHist;
    std::vector<double> renHist;
    std::vector<tui::Box> lastBoxes;
    double lastMetricsTs = -1.0;
    float lastFrameW = 0.f;
    float lastFrameH = 0.f;
    bool haveMetrics = false;
    bool metricsStale = false;
    const size_t kHistMax = 120;
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
                         (dir / "detector_tui.log").string(), err)) {
            message = "launch failed: " + err;
            return;
        }
        running = true;
        everRan = true;
        stopDeadline = -1.0;
        started = std::chrono::steady_clock::now();
        message = "running " + cfg.file;
        runningFile = cfg.file;

        // Fresh run: drop the previous run's charts and box snapshot.
        capHist.clear();
        detHist.clear();
        renHist.clear();
        lastBoxes.clear();
        lastMetricsTs = -1.0;
        lastFrameW = lastFrameH = 0.f;
        haveMetrics = false;
        metricsStale = false;
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

    // Lower pane while the detector runs with fresh metrics: latency
    // sparklines, the ASCII box map around the crosshair, the current numbers
    // and a short log tail so errors stay visible.
    auto drawMetrics = [&](Canvas &cv, int top, int msgRow, double up, double detCount,
                           const std::string &model, const std::string &prec,
                           bool graph) {
        const int avail = msgRow - top;
        if (avail < 6) return;

        const int logTail = (avail >= 14) ? 3 : 2;
        const int statsRows = (avail - logTail >= 9) ? 2 : 1;
        const int graphs = (avail - logTail - statsRows >= 8) ? 4 : 0; // 3 rows + spacer
        int boxRows = avail - logTail - statsRows - graphs - 1;        // 1 = box header
        if (boxRows < 2) boxRows = 2;
        if (boxRows > 16) boxRows = 16;

        const int logTailTop = msgRow - logTail;
        int row = top;

        if (graphs > 0) {
            const int labelW = 8;
            int sparkW = cv.cols - labelW - 24;
            if (sparkW < 10) sparkW = 10;
            struct Series {
                const char *name;
                const std::vector<double> *h;
            };
            const Series series[3] = {{ "capture", &capHist },
                                      { "detect", &detHist },
                                      { "render", &renHist }};
            for (int i = 0; i < 3; ++i) {
                const std::vector<double> &h = *series[i].h;
                double sum = 0;
                for (double v : h) sum += v;
                const double last = h.empty() ? 0.0 : h.back();
                const double mean = h.empty() ? 0.0 : sum / static_cast<double>(h.size());
                std::string line = std::string(" ") + series[i].name;
                while (line.size() < static_cast<size_t>(labelW)) line += ' ';
                line += tui::sparkline(h, sparkW);
                line += "  last " + fmt2(last) + "  avg " + fmt2(mean);
                cv.put(row, 1, trunc(line, static_cast<size_t>(cv.cols)));
                ++row;
            }
            ++row; // spacer between the graphs and the box map
        }

        cv.put(row, 1, "Box map (capture frame around the crosshair)");
        cv.setStyle(row, STYLE_DIM);
        ++row;

        const int mapTop = row;
        if (!lastBoxes.empty() && lastFrameW > 0.f && lastFrameH > 0.f) {
            const int cols = std::min(31, cv.cols - 2);
            int mapRows = logTailTop - mapTop - statsRows;
            if (mapRows > boxRows) mapRows = boxRows;
            if (mapRows > 0) {
                std::vector<std::string> map =
                    tui::renderBoxMap(lastBoxes, lastFrameW, lastFrameH, cols, mapRows);
                for (size_t i = 0; i < map.size(); ++i) {
                    cv.put(mapTop + static_cast<int>(i), 1, map[i]);
                }
            }
        } else if (mapTop < logTailTop) {
            cv.put(mapTop, 1, "  no box data yet (older detector, or nothing detected)");
            cv.setStyle(mapTop, STYLE_DIM);
        }
        row = std::min(mapTop + boxRows, logTailTop);

        // Stats: the running numbers, then the actionable target line.
        const std::string s1 =
            "  up " + std::to_string(static_cast<int>(up)) + "s   det/frame " +
            fmt1(detCount) + "   model " + (model.empty() ? "?" : model) + "   " +
            (prec.empty() ? "?" : prec) + "   graph " + (graph ? "yes" : "no") +
            "   samples " + std::to_string(capHist.size());
        std::string s2 = "  no boxes to measure";
        if (!lastBoxes.empty() && lastFrameW > 0.f && lastFrameH > 0.f) {
            s2 = "  nearest box " +
                 fmt2(tui::nearestBoxDistance(lastBoxes, lastFrameW, lastFrameH)) +
                 (tui::crosshairInsideBox(lastBoxes, lastFrameW, lastFrameH)
                      ? "   ON TARGET"
                      : "");
        }
        if (statsRows >= 2 && row + 1 < logTailTop) {
            cv.put(row++, 1, trunc(s1, static_cast<size_t>(cv.cols)));
            cv.put(row++, 1, trunc(s2, static_cast<size_t>(cv.cols)));
        } else if (row < logTailTop) {
            cv.put(row++, 1, trunc(s1 + "   " + s2, static_cast<size_t>(cv.cols)));
        }

        // Log tail, pinned to the bottom of the pane.
        std::string logData = readFileTail(dir / "detector_tui.log", 8192);
        std::vector<std::string> lines = splitLines(logData);
        int from = static_cast<int>(lines.size()) - logTail;
        if (from < 0) from = 0;
        for (int i = from; i < static_cast<int>(lines.size()); ++i) {
            const int r = logTailTop + (i - from);
            if (r < 1 || r >= msgRow) break;
            cv.put(r, 1,
                   trunc(sanitize(lines[static_cast<size_t>(i)]),
                         static_cast<size_t>(cv.cols)));
        }
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
            message = "detector exited (code " + std::to_string(lastExit) + ")";
        }
        if (running && stopDeadline > 0.0 && nowSec() >= stopDeadline) {
            child.kill();
            running = false;
            stopDeadline = -1.0;
            lastExit = child.exitCode();
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

        // Shared with the lower pane: current numbers + graph history.
        double mUp = 0;
        double mDet = 0;
        std::string mModel;
        std::string mPrec;
        bool mGraph = false;
        bool mFresh = false;

        if (running) {
            double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
            info.push_back("Detector: RUNNING  pid " + std::to_string(child.pid()) +
                           "  up " + std::to_string(static_cast<int>(elapsed)) + "s");

            std::string sj = readFileTail(dir / "status.json", 8192);
            if (sj.empty()) {
                info.push_back("  metrics: waiting for the first status.json ...");
            } else {
                double ts = 0;
                bool fresh = jsonNumber(sj, "ts", ts) &&
                             (std::time(nullptr) - static_cast<time_t>(ts)) <= 3;
                std::string cap = jsonObject(sj, "capture");
                std::string det = jsonObject(sj, "detect");
                std::string ren = jsonObject(sj, "render");
                double capV = 0, detV = 0, renV = 0;
                const bool haveCap = jsonNumber(cap, "avg", capV);
                const bool haveDet = jsonNumber(det, "avg", detV);
                const bool haveRen = jsonNumber(ren, "avg", renV);
                std::string capAvg = haveCap ? fmt2(capV) : "?";
                std::string detAvg = haveDet ? fmt2(detV) : "?";
                std::string renAvg = haveRen ? fmt2(renV) : "?";
                info.push_back("  capture " + capAvg + " ms   detect " + detAvg +
                               " ms   render " + renAvg + " ms" + (fresh ? "" : "   (stale)"));
                std::string model, prec;
                double dcount = 0, up = 0;
                bool graph = false;
                jsonString(sj, "model", model);
                jsonString(sj, "precision", prec);
                jsonNumber(sj, "detections", dcount);
                jsonNumber(sj, "uptime_s", up);
                jsonBool(sj, "graph", graph);
                mModel = model;
                mPrec = prec;
                mDet = dcount;
                mUp = up;
                mGraph = graph;
                mFresh = fresh;
                info.push_back("  detections " + fmt1(dcount) + "   model " +
                               (model.empty() ? "?" : model) + "   " +
                               (prec.empty() ? "?" : prec) + "   graph " +
                               (graph ? "yes" : "no") + "   uptime " +
                               std::to_string(static_cast<int>(up)) + "s");

                if (fresh && ts != lastMetricsTs) {
                    // New sample: extend the graphs, refresh the box snapshot.
                    lastMetricsTs = ts;
                    if (haveCap) capHist.push_back(capV);
                    if (haveDet) detHist.push_back(detV);
                    if (haveRen) renHist.push_back(renV);
                    while (capHist.size() > kHistMax) capHist.erase(capHist.begin());
                    while (detHist.size() > kHistMax) detHist.erase(detHist.begin());
                    while (renHist.size() > kHistMax) renHist.erase(renHist.begin());
                    haveMetrics = true;

                    double fw = 0, fh = 0;
                    if (jsonNumber(sj, "fw", fw) && jsonNumber(sj, "fh", fh) && fw > 0 &&
                        fh > 0) {
                        lastFrameW = static_cast<float>(fw);
                        lastFrameH = static_cast<float>(fh);
                        lastBoxes = jsonBoxArray(sj);
                    } else {
                        lastFrameW = lastFrameH = 0.f;
                        lastBoxes.clear();
                    }
                }

                metricsStale = haveMetrics &&
                               (std::time(nullptr) - static_cast<time_t>(ts)) > 5.0;
                if (metricsStale) {
                    message = "metrics stale - no status.json update for over 5s";
                } else if (message.rfind("metrics stale", 0) == 0) {
                    message = "running " + runningFile;
                }
            }
            info.push_back("");
            info.push_back("  s stops the detector, q quits the TUI");
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

        // Lower pane spans the full width below the top section: live metrics
        // once status.json is flowing, the rolling log otherwise.
        const int paneHeaderRow = 5 + topBlock;
        const int paneTop = paneHeaderRow + 1;
        const int msgRow = R - 1;

        const bool showMetrics = running && haveMetrics && (msgRow - paneTop) >= 6;
        if (showMetrics) {
            std::string head = "Metrics (status.json)   capture/detect/render ms";
            if (metricsStale) head += "   [stale]";
            cv.put(paneHeaderRow, 1, trunc(head, static_cast<size_t>(C)));
        } else if (running) {
            const bool metricsOff = sel < static_cast<int>(configs.size()) &&
                                    configs[static_cast<size_t>(sel)].metrics.empty();
            cv.put(paneHeaderRow, 1,
                   trunc(metricsOff ? "Log (metrics off: set MetricsStatus in the config)"
                                    : "Log (starting - waiting for metrics)",
                         static_cast<size_t>(C)));
        } else {
            cv.put(paneHeaderRow, 1,
                   "Log (" + (dir / "detector_tui.log").filename().string() + ")");
        }
        cv.setStyle(paneHeaderRow, STYLE_SECTION);

        if (showMetrics) {
            drawMetrics(cv, paneTop, msgRow, mUp, mDet, mModel, mPrec, mGraph);
        } else {
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
            keys = " running:  s stop (clean)   q quit (stops detector)   Ctrl+C quit";
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
