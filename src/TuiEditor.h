#pragma once
// TuiEditor — raw-line INI editor for the detector TUI.
//
// Edits one value at a time. Everything else in the file — comments, blank
// lines, section headers, key order, even the odd whitespace around the '='
// — survives a save byte-for-byte: only the byte range of the edited value is
// replaced. Saves are atomic (write <path>.tmp, then rename), the same pattern
// the detector uses for status.json.
//
// ASCII only, C++17, header-only.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "TuiTerminal.h"

namespace tui {

enum class EditorAction {
    None,
    Back,
    SaveAndRun,
    Quit
};

class IniEditor {
public:
    // One "key = value" line. lineIndex points into the raw line array; the
    // value occupies [valStart, valEnd) of that line, so save() can splice the
    // new text in without touching anything else.
    struct Row {
        long long lineIndex = -1;
        std::string key;
        std::string value;
        size_t valStart = 0;
        size_t valEnd = 0;
        bool hasComment = false;
        size_t commentStart = 0;
    };

    bool load(const std::string &path, std::string &err) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            err = "cannot open " + path;
            return false;
        }
        std::string data((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

        m_lines.clear();
        std::string cur;
        for (char c : data) {
            if (c == '\n') {
                m_lines.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        m_lines.push_back(cur); // trailing piece (possibly empty)

        m_rows.clear();
        for (size_t i = 0; i < m_lines.size(); ++i) parseRow(i);

        m_path = path;
        m_cursor = 0;
        m_top = 0;
        m_dirty = false;
        m_editing = false;
        m_confirm = false;
        m_editBuffer.clear();
        m_editCursor = 0;
        err.clear();
        return true;
    }

    bool save(std::string &err) {
        std::string data;
        for (size_t i = 0; i < m_lines.size(); ++i) {
            if (i) data += '\n';
            data += m_lines[i];
        }
        const std::string tmp = m_path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) {
                err = "cannot write " + tmp;
                return false;
            }
            f.write(data.data(), static_cast<std::streamsize>(data.size()));
            f.flush();
            if (!f) {
                err = "write failed: " + tmp;
                return false;
            }
        }
        if (std::rename(tmp.c_str(), m_path.c_str()) != 0) {
            err = "rename failed: " + tmp;
            std::remove(tmp.c_str());
            return false;
        }
        m_dirty = false;
        err.clear();
        return true;
    }

    const std::string &path() const { return m_path; }
    bool dirty() const { return m_dirty; }
    bool editing() const { return m_editing; }
    int cursor() const { return m_cursor; }
    const std::string &editBuffer() const { return m_editBuffer; }
    int editCursor() const { return m_editCursor; }
    size_t rowCount() const { return m_rows.size(); }

    // Footer hint for the current mode.
    std::string footer() const {
        if (m_confirm) {
            return " unsaved changes:  s save   d discard   esc cancel";
        }
        if (m_editing) {
            return " enter commit   esc cancel   left/right move   backspace delete";
        }
        return " up/down move   enter edit   s save   r save+run   esc back";
    }

    // Keeps the cursor row inside a window of `visibleRows` lines.
    void ensureVisible(int visibleRows) {
        if (visibleRows <= 0) {
            m_top = 0;
            return;
        }
        if (m_cursor < m_top) m_top = m_cursor;
        if (m_cursor >= m_top + visibleRows) m_top = m_cursor - visibleRows + 1;
        if (m_top < 0) m_top = 0;
        int maxTop = static_cast<int>(m_rows.size()) - visibleRows;
        if (maxTop < 0) maxTop = 0;
        if (m_top > maxTop) m_top = maxTop;
    }

    // Rows for the window around the cursor: {rowIndex, text}. While a value is
    // being edited its row shows the buffer (with a '|' caret) instead.
    std::vector<std::pair<int, std::string>> visibleLines(int visibleRows) const {
        std::vector<std::pair<int, std::string>> out;
        if (visibleRows <= 0) return out;
        int first = m_top;
        if (first < 0) first = 0;
        for (int i = first;
             i < static_cast<int>(m_rows.size()) &&
             static_cast<int>(out.size()) < visibleRows;
             ++i) {
            const Row &r = m_rows[static_cast<size_t>(i)];
            const std::string &ln = m_lines[static_cast<size_t>(r.lineIndex)];
            std::string text;
            if (m_editing && i == m_cursor) {
                std::string buf = m_editBuffer;
                size_t c = std::min(static_cast<size_t>(std::max(0, m_editCursor)), buf.size());
                buf.insert(buf.begin() + static_cast<long>(c), '|');
                text = ln.substr(0, r.valStart) + buf + ln.substr(r.valEnd);
            } else {
                text = ln;
            }
            out.push_back({i, text});
        }
        return out;
    }

    EditorAction handleKey(const KeyEvent &k, std::string &msg) {
        if (m_rows.empty()) {
            if (k.key == Key::Escape || (k.key == Key::Char && lowerCh(k.ch) == 'q')) {
                return EditorAction::Back;
            }
            return EditorAction::None;
        }

        if (m_confirm) {
            if (k.key == Key::Char && lowerCh(k.ch) == 's') {
                std::string err;
                if (save(err)) {
                    msg = "saved " + m_path;
                } else {
                    msg = "save failed: " + err;
                }
                m_confirm = false;
                return EditorAction::Back;
            }
            if (k.key == Key::Char && lowerCh(k.ch) == 'd') {
                m_dirty = false;
                m_confirm = false;
                msg = "changes discarded";
                return EditorAction::Back;
            }
            if (k.key == Key::Escape) {
                m_confirm = false;
            }
            return EditorAction::None;
        }

        if (m_editing) {
            switch (k.key) {
                case Key::Escape:
                    m_editing = false;
                    return EditorAction::None;
                case Key::Enter: {
                    const std::string v = trim(m_editBuffer);
                    spliceValue(static_cast<size_t>(m_cursor), v);
                    m_dirty = true;
                    m_editing = false;
                    msg = "edited " + m_rows[static_cast<size_t>(m_cursor)].key;
                    return EditorAction::None;
                }
                case Key::Backspace:
                    if (m_editCursor > 0) {
                        m_editBuffer.erase(static_cast<size_t>(m_editCursor - 1), 1);
                        --m_editCursor;
                    }
                    return EditorAction::None;
                case Key::Left:
                    if (m_editCursor > 0) --m_editCursor;
                    return EditorAction::None;
                case Key::Right:
                    if (m_editCursor < static_cast<int>(m_editBuffer.size())) ++m_editCursor;
                    return EditorAction::None;
                case Key::Char:
                    m_editBuffer.insert(m_editBuffer.begin() + m_editCursor, k.ch);
                    ++m_editCursor;
                    return EditorAction::None;
                default:
                    return EditorAction::None;
            }
        }

        // Browse.
        if (k.key == Key::Up || (k.key == Key::Char && lowerCh(k.ch) == 'k')) {
            moveCursor(-1);
        } else if (k.key == Key::Down || (k.key == Key::Char && lowerCh(k.ch) == 'j')) {
            moveCursor(1);
        } else if (k.key == Key::Enter) {
            m_editing = true;
            m_editBuffer = m_rows[static_cast<size_t>(m_cursor)].value;
            m_editCursor = static_cast<int>(m_editBuffer.size());
        } else if (k.key == Key::Char && lowerCh(k.ch) == 's') {
            std::string err;
            if (save(err)) {
                msg = "saved " + m_path;
            } else {
                msg = "save failed: " + err;
            }
        } else if (k.key == Key::Char && lowerCh(k.ch) == 'r') {
            std::string err;
            if (!save(err)) {
                msg = "save failed: " + err;
                return EditorAction::None;
            }
            msg = "saved " + m_path + " - starting detector";
            return EditorAction::SaveAndRun;
        } else if (k.key == Key::Escape || (k.key == Key::Char && lowerCh(k.ch) == 'q')) {
            if (m_dirty) {
                m_confirm = true;
            } else {
                return EditorAction::Back;
            }
        }
        return EditorAction::None;
    }

private:
    static char lowerCh(char c) {
        return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    }

    static bool spaceCh(char c) {
        return c == ' ' || c == '\t' || c == '\r';
    }

    static std::string trim(const std::string &s) {
        size_t b = 0;
        size_t e = s.size();
        while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
        while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
        return s.substr(b, e - b);
    }

    void parseRow(size_t i) {
        const std::string &ln = m_lines[i];
        const size_t eq = ln.find('=');
        if (eq == std::string::npos) return;
        const std::string key = trim(ln.substr(0, eq));
        if (key.empty()) return;
        if (key[0] == '#' || key[0] == ';') return; // commented-out line

        size_t v = eq + 1;
        while (v < ln.size() && (ln[v] == ' ' || ln[v] == '\t')) ++v;
        size_t e = ln.size();
        while (e > v && spaceCh(ln[e - 1])) --e;

        // A trailing " #..." / " ;..." comment stays with the line.
        size_t cpos = std::string::npos;
        for (size_t p = v + 1; p < e; ++p) {
            if ((ln[p] == '#' || ln[p] == ';') && spaceCh(ln[p - 1])) {
                cpos = p;
                break;
            }
        }
        size_t vend = (cpos == std::string::npos) ? e : cpos;
        while (vend > v && (ln[vend - 1] == ' ' || ln[vend - 1] == '\t')) --vend;

        Row r;
        r.lineIndex = static_cast<long long>(i);
        r.key = key;
        r.valStart = v;
        r.valEnd = vend;
        r.value = ln.substr(v, vend - v);
        if (cpos != std::string::npos) {
            r.hasComment = true;
            r.commentStart = cpos;
        }
        m_rows.push_back(r);
    }

    void spliceValue(size_t rowIdx, const std::string &newValue) {
        Row &r = m_rows[rowIdx];
        const size_t li = static_cast<size_t>(r.lineIndex);
        const std::string &ln = m_lines[li];
        const size_t oldLen = r.valEnd - r.valStart;
        m_lines[li] = ln.substr(0, r.valStart) + newValue + ln.substr(r.valEnd);
        r.value = newValue;
        r.valEnd = r.valStart + newValue.size();
        if (r.hasComment) {
            r.commentStart = r.commentStart + newValue.size() - oldLen;
        }
    }

    void moveCursor(int delta) {
        const int n = static_cast<int>(m_rows.size());
        if (n == 0) return;
        m_cursor += delta;
        if (m_cursor < 0) m_cursor = n - 1;
        if (m_cursor >= n) m_cursor = 0;
    }

    std::vector<std::string> m_lines;
    std::vector<Row> m_rows;
    std::string m_path;
    int m_cursor = 0;
    int m_top = 0;
    bool m_dirty = false;
    bool m_editing = false;
    bool m_confirm = false;
    std::string m_editBuffer;
    int m_editCursor = 0;
};

} // namespace tui
