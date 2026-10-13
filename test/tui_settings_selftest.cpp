// Minimal self-check for the TUI settings helpers. No framework: prints
// FAIL and exits non-zero on the first-failing batch, so it can be run from
// ctest or directly. Covers TuiSettings.h parsing/editing helpers and the
// IniEditor getValue/setValue splice round-trip.
#include "TuiCharts.h"
#include "TuiEditor.h"
#include "TuiSettings.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++failures; } } while (0)

static const tui::Field &fieldByKey(const char *key) {
    for (const tui::Field &f : tui::settingsFields())
        if (std::strcmp(f.key, key) == 0) return f;
    static const tui::Field none{};
    return none;
}

int main() {
    double d = 0;
    CHECK(tui::parseDouble("3.5", d) && d == 3.5, "parseDouble 3.5");
    CHECK(tui::parseDouble("1e2", d) && d == 100.0, "parseDouble exponent");
    CHECK(!tui::parseDouble("3.5x", d), "parseDouble rejects trailing garbage");
    CHECK(!tui::parseDouble("", d), "parseDouble rejects empty");

    long long l = 0;
    CHECK(tui::parseLong("42", l) && l == 42, "parseLong 42");
    CHECK(tui::parseLong("042", l) && l == 42, "parseLong leading zero");
    CHECK(!tui::parseLong("4.2", l), "parseLong rejects float");
    CHECK(!tui::parseLong("0x10", l), "parseLong rejects hex");

    bool b = false;
    CHECK(tui::parseBool("true", b) && b, "parseBool true");
    CHECK(tui::parseBool("OFF", b) && !b, "parseBool OFF");
    CHECK(!tui::parseBool("yes please", b), "parseBool rejects junk");

    CHECK(tui::trim("  hi \r\n") == "hi", "trim");

    const tui::Field &mv = fieldByKey("ModelVersion");
    const std::vector<std::string> cv = tui::choiceValues(mv);
    CHECK(cv.size() == 4 && cv[0] == "auto" && cv[3] == "v26", "choiceValues ModelVersion");

    const tui::Field &smooth = fieldByKey("Smoothing");
    CHECK(tui::normalizeInt(smooth, "7") == "7", "normalizeInt 7");
    CHECK(tui::normalizeInt(smooth, "99").empty(), "normalizeInt rejects out of range");
    const tui::Field &fps = fieldByKey("CaptureFPS");
    CHECK(tui::normalizeInt(fps, "077") == "77", "normalizeInt strips leading zero");
    CHECK(tui::normalizeFloat(smooth, "4.5") == "4.5", "normalizeFloat 4.5");
    CHECK(tui::normalizeFloat(smooth, "0").empty(), "normalizeFloat rejects below min");

    // TuiCharts visuals
    const std::vector<std::string> radar = tui::renderAimCircle(20, 9, 106.0, 45.0);
    CHECK(radar.size() == 9 && radar[0].size() == 20, "renderAimCircle dims");
    CHECK(radar[4][10] == '+', "renderAimCircle crosshair at centre");
    const std::string range = tui::renderRangeBar(20, 0.25, 0.65, 0.0, 1.0);
    CHECK(range.size() == 20 && range.find('[') != std::string::npos &&
          range.find(']') != std::string::npos && range.find('#') != std::string::npos,
          "renderRangeBar band");
    const std::string bar = tui::renderProgressBar(20, 5.0, 1.0, 10.0);
    CHECK(bar.size() == 20 && bar[0] == '[' && bar[19] == ']' && bar.find('=') != std::string::npos,
          "renderProgressBar fill");

    tui::LineEdit le;
    le.start("ab");
    tui::KeyEvent k;
    k.key = tui::Key::Char;
    k.ch = 'c';
    CHECK(le.handle(k) && le.buf == "abc" && le.pos == 3, "LineEdit char");
    k.key = tui::Key::Home;
    CHECK(le.handle(k) && le.pos == 0, "LineEdit home");
    k.key = tui::Key::Delete;
    CHECK(le.handle(k) && le.buf == "bc" && le.pos == 0, "LineEdit delete");
    k.key = tui::Key::Enter;
    CHECK(!le.handle(k), "LineEdit passes through unknown keys");

    // IniEditor setValue/getValue with comment + sibling-key preservation.
    const char *tmp = "/tmp/tui_selftest.ini";
    {
        std::FILE *f = std::fopen(tmp, "wb");
        std::fputs("Smoothing = 4   # 1-10 knob\nOther = x\n", f);
        std::fclose(f);
    }
    tui::IniEditor ed;
    std::string err;
    CHECK(ed.load(tmp, err), "IniEditor load");
    CHECK(ed.getValue("Smoothing") == "4", "getValue");
    CHECK(ed.setValue("Smoothing", "7"), "setValue");
    CHECK(ed.getValue("Smoothing") == "7", "getValue after set");
    CHECK(ed.dirty(), "dirty after set");
    CHECK(ed.save(err), "save");

    tui::IniEditor ed2;
    CHECK(ed2.load(tmp, err), "reload");
    CHECK(ed2.getValue("Smoothing") == "7", "value persisted");
    std::FILE *f = std::fopen(tmp, "rb");
    char buf[64] = {0};
    std::fread(buf, 1, 63, f);
    std::fclose(f);
    CHECK(std::strstr(buf, "# 1-10 knob") != nullptr, "comment preserved");
    CHECK(std::strstr(buf, "Other = x") != nullptr, "sibling key preserved");

    std::remove(tmp);
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
