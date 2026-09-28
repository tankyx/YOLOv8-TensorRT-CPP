#pragma once
// TuiTerminal — minimal cross-platform terminal control for the detector TUI.
//
// POSIX: termios raw mode, ANSI escape sequences, select() for timed reads.
// Windows: console input/output modes with ENABLE_VIRTUAL_TERMINAL_PROCESSING
// (Windows 10 1511+ / Windows Terminal), ReadConsoleInputW for key events so
// arrow keys work without relying on VT input translation.
//
// Keys are normalized to a small enum; both arrow keys and j/k style vim keys
// are available to callers. Character events keep their original case, so
// text-entry screens (the config editor) can type values verbatim; callers
// comparing command keys normalize with a tolower of their own. The UI itself
// is ASCII-only on purpose — no codepage assumptions on the Windows console.

#include <cstdint>
#include <cstdio>
#include <string>

#if defined(_WIN32)
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0A00   // Virtual terminal processing needs Win10.
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <sys/ioctl.h>
#  include <sys/select.h>
#  include <termios.h>
#  include <unistd.h>
#endif

namespace tui {

enum class Key {
    None,
    Up,
    Down,
    Left,
    Right,
    Enter,
    Escape,
    Backspace,
    Resize,
    Char,
    Unknown
};

struct KeyEvent {
    Key key = Key::None;
    char ch = 0; // valid when key == Key::Char; original case is preserved
};

struct Size {
    int rows = 24;
    int cols = 80;
};

class Terminal {
public:
    bool init() {
#if defined(_WIN32)
        m_hIn = GetStdHandle(STD_INPUT_HANDLE);
        m_hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (m_hIn == INVALID_HANDLE_VALUE || m_hOut == INVALID_HANDLE_VALUE) {
            return false;
        }
        if (!GetConsoleMode(m_hIn, &m_savedInMode) ||
            !GetConsoleMode(m_hOut, &m_savedOutMode)) {
            return false; // redirected stdin/stdout: not an interactive terminal.
        }
        m_savedInCP = GetConsoleCP();
        m_savedOutCP = GetConsoleOutputCP();

        // Keep ENABLE_PROCESSED_INPUT so Ctrl+C still raises a console control
        // event (handled by the app); drop line input/echo so we get raw keys.
        DWORD inMode = m_savedInMode & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT);
        inMode |= ENABLE_EXTENDED_FLAGS | ENABLE_WINDOW_INPUT;
        inMode &= ~ENABLE_QUICK_EDIT_MODE; // quick-edit would freeze input on click
        SetConsoleMode(m_hIn, inMode);

        DWORD outMode = m_savedOutMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING |
                        ENABLE_PROCESSED_OUTPUT;
        SetConsoleMode(m_hOut, outMode);

        SetConsoleCP(CP_UTF8);
        SetConsoleOutputCP(CP_UTF8);

        m_tty = true;
        write("\x1b[?1049h"); // alternate screen buffer
        hideCursor();
        clear();
        return true;
#else
        if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) {
            return false;
        }
        if (tcgetattr(STDIN_FILENO, &m_saved) != 0) {
            return false;
        }
        m_savedOk = true;
        struct termios raw = m_saved;
        raw.c_lflag &= ~(ICANON | ECHO);
        raw.c_iflag &= ~(ICRNL);
        raw.c_cc[VMIN] = 1;
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        m_tty = true;
        write("\x1b[?1049h");
        hideCursor();
        clear();
        return true;
#endif
    }

    void restore() {
        if (!m_tty) return;
        showCursor();
        write("\x1b[0m");
        write("\x1b[?1049l"); // back to the normal screen buffer
        flush();
#if defined(_WIN32)
        SetConsoleMode(m_hIn, m_savedInMode);
        SetConsoleMode(m_hOut, m_savedOutMode);
        SetConsoleCP(m_savedInCP);
        SetConsoleOutputCP(m_savedOutCP);
#else
        if (m_savedOk) {
            tcsetattr(STDIN_FILENO, TCSANOW, &m_saved);
        }
#endif
        m_tty = false;
    }

    bool isTerminal() const { return m_tty; }

    Size size() const {
#if defined(_WIN32)
        CONSOLE_SCREEN_BUFFER_INFO info{};
        if (GetConsoleScreenBufferInfo(m_hOut, &info)) {
            Size s;
            s.cols = info.srWindow.Right - info.srWindow.Left + 1;
            s.rows = info.srWindow.Bottom - info.srWindow.Top + 1;
            return s;
        }
        return Size{};
#else
        struct winsize ws{};
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
            Size s;
            s.cols = ws.ws_col;
            s.rows = ws.ws_row;
            return s;
        }
        return Size{};
#endif
    }

    void write(const std::string &s) {
#if defined(_WIN32)
        DWORD written = 0;
        WriteFile(m_hOut, s.data(), static_cast<DWORD>(s.size()), &written, nullptr);
#else
        std::fwrite(s.data(), 1, s.size(), stdout);
#endif
    }

    void flush() {
#if defined(_WIN32)
        // WriteFile is unbuffered for console handles; nothing to do.
#else
        std::fflush(stdout);
#endif
    }

    void moveTo(int row, int col) { // 1-based
        char buf[32];
        std::snprintf(buf, sizeof(buf), "\x1b[%d;%dH", row, col);
        write(buf);
    }

    void clear() { write("\x1b[2J\x1b[H"); }
    void hideCursor() { write("\x1b[?25l"); }
    void showCursor() { write("\x1b[?25h"); }

    // Waits up to timeoutMs for a key. Returns {Key::None} on timeout.
    KeyEvent readKey(int timeoutMs) {
#if defined(_WIN32)
        DWORD wait = WaitForSingleObject(m_hIn, static_cast<DWORD>(timeoutMs < 0 ? 0 : timeoutMs));
        if (wait != WAIT_OBJECT_0) {
            return KeyEvent{};
        }
        INPUT_RECORD recs[32];
        DWORD n = 0;
        if (!ReadConsoleInputW(m_hIn, recs, 32, &n)) {
            return KeyEvent{};
        }
        for (DWORD i = 0; i < n; ++i) {
            if (recs[i].EventType == WINDOW_BUFFER_SIZE_EVENT) {
                return KeyEvent{Key::Resize, 0};
            }
            if (recs[i].EventType != KEY_EVENT || !recs[i].Event.KeyEvent.bKeyDown) {
                continue;
            }
            const KEY_EVENT_RECORD &ke = recs[i].Event.KeyEvent;
            switch (ke.wVirtualKeyCode) {
                case VK_UP:    return KeyEvent{Key::Up, 0};
                case VK_DOWN:  return KeyEvent{Key::Down, 0};
                case VK_LEFT:  return KeyEvent{Key::Left, 0};
                case VK_RIGHT: return KeyEvent{Key::Right, 0};
                case VK_RETURN: return KeyEvent{Key::Enter, 0};
                case VK_ESCAPE: return KeyEvent{Key::Escape, 0};
                case VK_BACK:  return KeyEvent{Key::Backspace, 0};
                default: break;
            }
            wchar_t wc = ke.uChar.UnicodeChar;
            if (wc >= L' ' && wc < 127) {
                return KeyEvent{Key::Char, static_cast<char>(wc)};
            }
        }
        return KeyEvent{};
#else
        int b0 = nextByte(timeoutMs);
        if (b0 < 0) {
            return KeyEvent{};
        }
        if (b0 == 0x1b) {
            // A lone ESC may be the start of a split escape sequence (arrow
            // keys often arrive as ESC [ A across two reads on a pty). Wait
            // briefly for the remainder before deciding.
            const int b1 = nextByte(30);
            if (b1 == '[') {
                const int b2 = nextByte(30);
                switch (b2) {
                    case 'A': return KeyEvent{Key::Up, 0};
                    case 'B': return KeyEvent{Key::Down, 0};
                    case 'C': return KeyEvent{Key::Right, 0};
                    case 'D': return KeyEvent{Key::Left, 0};
                    default: break;
                }
            } else {
                ungetByte(b1); // not a sequence: hand the byte to the next read
            }
            return KeyEvent{Key::Escape, 0};
        }
        if (b0 == '\r' || b0 == '\n') {
            return KeyEvent{Key::Enter, 0};
        }
        if (b0 == 0x7f || b0 == 0x08) {
            return KeyEvent{Key::Backspace, 0};
        }
        if (b0 == 3) { // raw Ctrl+C (should normally arrive as SIGINT)
            return KeyEvent{Key::Char, 'c'};
        }
        if (b0 >= 32 && b0 < 127) {
            return KeyEvent{Key::Char, static_cast<char>(b0)};
        }
        return KeyEvent{Key::Unknown, 0};
#endif
    }

private:
#if !defined(_WIN32)
    // Next byte from the read-ahead buffer, refilling from stdin when empty.
    // Returns -1 on timeout or EOF. Reading more than one byte per syscall is
    // normal (fast typing, key repeat, paste), so every byte must be kept.
    int nextByte(int timeoutMs) {
        if (m_bufPos < m_bufLen) return m_buf[m_bufPos++];
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(STDIN_FILENO, &fds);
        struct timeval tv;
        tv.tv_sec = timeoutMs / 1000;
        tv.tv_usec = (timeoutMs % 1000) * 1000;
        if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) <= 0) return -1;
        const ssize_t n = read(STDIN_FILENO, m_buf, sizeof(m_buf));
        if (n <= 0) return -1;
        m_bufLen = static_cast<int>(n);
        m_bufPos = 1;
        return m_buf[0];
    }

    void ungetByte(int b) {
        if (b < 0) return;
        if (m_bufPos > 0) m_buf[--m_bufPos] = static_cast<unsigned char>(b);
    }
#endif

#if defined(_WIN32)
    HANDLE m_hIn = INVALID_HANDLE_VALUE;
    HANDLE m_hOut = INVALID_HANDLE_VALUE;
    DWORD m_savedInMode = 0;
    DWORD m_savedOutMode = 0;
    UINT m_savedInCP = 0;
    UINT m_savedOutCP = 0;
#else
    struct termios m_saved {};
    bool m_savedOk = false;
    unsigned char m_buf[64];
    int m_bufLen = 0;
    int m_bufPos = 0;
#endif
    bool m_tty = false;
};

} // namespace tui
