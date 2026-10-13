#pragma once
// TuiSettings — curated, typed schema for the detector config values, plus the
// small pure helpers the settings screen uses. Header-only, ASCII only, C++17,
// same conventions as the rest of the TUI. Anything not listed here is reached
// through the Advanced (raw INI) screen.

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "TuiTerminal.h"

namespace tui {

enum class FieldType { Bool, Choice, Int, Float, Text, OnnxPicker };

// One curated setting. `choices` is a comma-separated list for Choice fields.
// min/max bound Int/Float fields (inclusive).
struct Field {
    const char *section;
    const char *key;
    const char *label;
    const char *help;
    FieldType type;
    double min = 0.0;
    double max = 0.0;
    const char *choices = nullptr;
};

// The curated set, in display order. Keys and choices mirror the detector's
// INI parsing (see object_detection_image_mt.cpp): Precision is "half"/"float",
// MaxSpeed is an int, GameFOV/Smoothing/…/gains are floats.
inline const std::vector<Field> &settingsFields() {
    static const std::vector<Field> f = {
        {"Aim", "GameFOV", "Game FOV", "game field of view in degrees (1-179)", FieldType::Float, 1, 179},
        {"Aim", "AimFOV", "Aim FOV", "aim-assist cone FOV in degrees (1-179)", FieldType::Int, 1, 179},
        {"Aim", "MouseSensitivity", "Mouse sensitivity", "screen-pixel -> mouse-count multiplier", FieldType::Float, 0.001, 100},
        {"Aim", "MinGain", "Gain randomization min", "aim gain lower bound (0-1)", FieldType::Float, 0, 1},
        {"Aim", "MaxGain", "Gain randomization max", "aim gain upper bound (0-1)", FieldType::Float, 0, 1},
        {"Aim", "MaxSpeed", "Max speed", "max mouse speed", FieldType::Int, 0, 1000},
        {"Aim", "Smoothing", "Smoothing", "1 = snappy, 10 = very smooth", FieldType::Float, 1, 10},

        {"Target", "HeadLabelID1", "Target class 1", "primary detection class id", FieldType::Int, 0, 999},
        {"Target", "HeadLabelID2", "Target class 2", "secondary detection class id", FieldType::Int, 0, 999},

        {"Overlay", "DebugView", "Overlay", "draw the detection overlay", FieldType::Bool},
        {"Overlay", "DebugOverlayTargetProcess", "Overlay process", "process name the overlay attaches to", FieldType::Text},
        {"Overlay", "DebugCaptureWindow", "Capture window", "show the capture-ROI debug window", FieldType::Bool},
        {"Overlay", "DebugCaptureWindowFPS", "Capture window FPS", "max fps of the debug window (0 = every frame)", FieldType::Int, 0, 240},
        {"Overlay", "DebugAimEnabled", "RMB snap aim", "hold RMB to engage unsmoothed snap aim", FieldType::Bool},
        {"Overlay", "DebugSnapGain", "Snap gain", "RMB snap screen-pixel multiplier", FieldType::Float, 0, 100},
        {"Overlay", "UseDirectGpuCapture", "Direct GPU capture", "DXGI/CUDA interop capture (no CPU copy)", FieldType::Bool},

        {"Capture", "CaptureWidth", "Capture width", "capture ROI width (px)", FieldType::Int, 64, 3840},
        {"Capture", "CaptureHeight", "Capture height", "capture ROI height (px)", FieldType::Int, 64, 2160},
        {"Capture", "CaptureFPS", "Capture FPS", "capture frame rate", FieldType::Int, 1, 500},
        {"Capture", "TrackCrosshair", "Track crosshair", "CPU template-match crosshair tracking", FieldType::Bool},

        {"Model", "ModelVersion", "Model version", "auto / v8 / v11 / v26", FieldType::Choice, 0, 0, "auto,v8,v11,v26"},
        {"Model", "Precision", "Precision", "half (fast) or float (exact)", FieldType::Choice, 0, 0, "half,float"},
        {"Model", "ModelPath", "Model path", "pick a .onnx model from the config dir", FieldType::OnnxPicker},

        {"HID", "CPI", "Mouse CPI", "mouse counts per inch", FieldType::Int, 100, 100000},
        {"HID", "HidVendorId", "HID vendor id", "hex, e.g. 0x3367", FieldType::Text},
        {"HID", "HidProductId", "HID product id", "hex, e.g. 0x1978", FieldType::Text},
        {"HID", "HidSerial", "HID serial", "optional serial filter (empty = any)", FieldType::Text},
    };
    return f;
}

// --- small pure helpers (exercised by the self-check) ---

inline std::string trim(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && static_cast<unsigned char>(s[b]) <= ' ') ++b;
    while (e > b && static_cast<unsigned char>(s[e - 1]) <= ' ') --e;
    return s.substr(b, e - b);
}

// Strict full-string parse (no trailing garbage).
inline bool parseDouble(const std::string &s, double &out) {
    if (s.empty()) return false;
    char *end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

inline bool parseLong(const std::string &s, long long &out) {
    if (s.empty()) return false;
    char *end = nullptr;
    out = std::strtoll(s.c_str(), &end, 10);
    return end != s.c_str() && *end == '\0';
}

inline bool parseBool(const std::string &s, bool &out) {
    std::string v;
    for (char c : s) v += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    if (v == "true" || v == "1" || v == "yes" || v == "on") { out = true; return true; }
    if (v == "false" || v == "0" || v == "no" || v == "off") { out = false; return true; }
    return false;
}

// Values for a Choice field, split on ','.
inline std::vector<std::string> choiceValues(const Field &f) {
    std::vector<std::string> out;
    if (!f.choices) return out;
    std::string cur;
    for (char c : std::string(f.choices)) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}

// Canonical int text, or empty when invalid or out of range.
inline std::string normalizeInt(const Field &f, const std::string &text) {
    long long v = 0;
    if (!parseLong(text, v)) return {};
    if (v < static_cast<long long>(f.min) || v > static_cast<long long>(f.max)) return {};
    return std::to_string(v);
}

// Float: returns the trimmed text unchanged (so the user's formatting is kept),
// or empty when invalid or out of range.
inline std::string normalizeFloat(const Field &f, const std::string &text) {
    double v = 0;
    if (!parseDouble(text, v)) return {};
    if (v < f.min || v > f.max) return {};
    return text;
}

// Minimal single-line edit buffer shared by the settings screen.
struct LineEdit {
    std::string buf;
    int pos = 0;

    void start(const std::string &s) { buf = s; pos = static_cast<int>(s.size()); }

    // Handles movement/typing keys. Returns false for keys it does not own
    // (the caller decides commit/cancel).
    bool handle(const KeyEvent &k) {
        switch (k.key) {
            case Key::Left:  if (pos > 0) --pos; return true;
            case Key::Right: if (pos < static_cast<int>(buf.size())) ++pos; return true;
            case Key::Home:  pos = 0; return true;
            case Key::End:   pos = static_cast<int>(buf.size()); return true;
            case Key::Backspace:
                if (pos > 0) { buf.erase(static_cast<size_t>(pos - 1), 1); --pos; }
                return true;
            case Key::Delete:
                if (pos < static_cast<int>(buf.size())) buf.erase(static_cast<size_t>(pos), 1);
                return true;
            case Key::Char:
                buf.insert(buf.begin() + pos, k.ch);
                ++pos;
                return true;
            default:
                return false;
        }
    }

    // The buffer with a '|' caret at the cursor.
    std::string withCaret() const {
        std::string s = buf;
        const size_t c = std::min(s.size(), static_cast<size_t>(std::max(0, pos)));
        s.insert(s.begin() + static_cast<long>(c), '|');
        return s;
    }
};

} // namespace tui
