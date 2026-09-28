#pragma once
// TuiProcess — child-process management for the detector TUI.
//
// Launches detect_object_image with stdout/stderr redirected to a log file
// (tailed by the UI), then polls for exit. Stopping is graceful first:
//   - POSIX:   SIGINT — the detector installs SIGINT/SIGTERM handlers and
//              shuts down cleanly.
//   - Windows: a synthetic Insert keypress — the detector polls
//              GetAsyncKeyState(VK_INSERT) in its main loop and exits through
//              its own shutdown path. Hard kill is the fallback in both cases.
//
// Orphan safety: the child dies with the TUI.
//   - POSIX:   prctl(PR_SET_PDEATHSIG, SIGTERM) in the forked child (Linux).
//   - Windows: the child is assigned to a job object with
//              JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.

#include <string>

#if defined(_WIN32)
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0A00
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <signal.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  if defined(__linux__)
#    include <sys/prctl.h>
#  endif
#endif

namespace tui {

class ChildProcess {
public:
    ChildProcess() = default;
    ~ChildProcess() {
        if (running()) {
            kill();
        }
        closeHandles();
    }

    ChildProcess(const ChildProcess &) = delete;
    ChildProcess &operator=(const ChildProcess &) = delete;

    // exe, workDir and logPath are filesystem paths; arg is a single argument
    // (the config filename). Returns false and fills err on failure.
    bool start(const std::string &exe, const std::string &arg,
               const std::string &workDir, const std::string &logPath,
               std::string &err) {
        if (running()) {
            err = "a detector process is already running";
            return false;
        }
        reset();

#if defined(_WIN32)
        std::wstring wExe = widen(exe);
        std::wstring wArg = widen(arg);
        std::wstring wDir = widen(workDir);
        std::wstring wLog = widen(logPath);

        if (GetFileAttributesW(wExe.c_str()) == INVALID_FILE_ATTRIBUTES) {
            err = "detector binary not found: " + exe;
            return false;
        }

        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;

        HANDLE hLog = CreateFileW(wLog.c_str(), GENERIC_WRITE,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                                  CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hLog == INVALID_HANDLE_VALUE) {
            err = "cannot create log file: " + logPath;
            return false;
        }
        HANDLE hNul = CreateFileW(L"NUL", GENERIC_READ,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                                  OPEN_EXISTING, 0, nullptr);

        STARTUPINFOW si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = (hNul == INVALID_HANDLE_VALUE) ? nullptr : hNul;
        si.hStdOutput = hLog;
        si.hStdError = hLog;

        std::wstring cmd = L"\"" + wExe + L"\" " + wArg;
        PROCESS_INFORMATION pi{};
        BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr,
                                 TRUE /* inherit handles */, 0, nullptr,
                                 wDir.empty() ? nullptr : wDir.c_str(), &si, &pi);
        CloseHandle(hLog);
        if (hNul != INVALID_HANDLE_VALUE) {
            CloseHandle(hNul);
        }
        if (!ok) {
            err = "CreateProcess failed (error " + std::to_string(GetLastError()) + ")";
            return false;
        }

        CloseHandle(pi.hThread);
        m_proc = pi.hProcess;
        m_pid = pi.dwProcessId;

        // Put the child in a kill-on-close job so it cannot outlive the TUI.
        m_job = CreateJobObjectW(nullptr, nullptr);
        if (m_job != nullptr) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(m_job, JobObjectExtendedLimitInformation,
                                    &info, sizeof(info));
            AssignProcessToJobObject(m_job, m_proc);
        }
        return true;
#else
        if (access(exe.c_str(), X_OK) != 0) {
            err = "detector binary not found or not executable: " + exe;
            return false;
        }
        pid_t pid = fork();
        if (pid < 0) {
            err = "fork failed";
            return false;
        }
        if (pid == 0) {
            // Child.
#if defined(__linux__)
            prctl(PR_SET_PDEATHSIG, SIGTERM); // die with the TUI
#endif
            if (!workDir.empty()) {
                (void)chdir(workDir.c_str());
            }
            int fd = open(logPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
            if (fd >= 0) {
                dup2(fd, STDOUT_FILENO);
                dup2(fd, STDERR_FILENO);
                if (fd > 2) close(fd);
            }
            int devnull = open("/dev/null", O_RDONLY);
            if (devnull >= 0) {
                dup2(devnull, STDIN_FILENO);
                if (devnull > 2) close(devnull);
            }
            execl(exe.c_str(), exe.c_str(), arg.c_str(), static_cast<char *>(nullptr));
            _exit(127); // exec failed; reported as exit code 127
        }
        m_pid = pid;
        return true;
#endif
    }

    // Non-blocking: true while the child is alive. Reaps the child when it exits.
    bool running() {
#if defined(_WIN32)
        if (m_proc == nullptr) return false;
        DWORD w = WaitForSingleObject(m_proc, 0);
        if (w == WAIT_OBJECT_0) {
            DWORD code = 0;
            GetExitCodeProcess(m_proc, &code);
            m_exitCode = static_cast<int>(code);
            finishWindowsStop();
            CloseHandle(m_proc);
            m_proc = nullptr;
            if (m_job) { CloseHandle(m_job); m_job = nullptr; }
            return false;
        }
        finishWindowsStopIfDue();
        return true;
#else
        if (m_pid <= 0) return false;
        int status = 0;
        pid_t r = waitpid(m_pid, &status, WNOHANG);
        if (r == m_pid) {
            m_exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            m_pid = -1;
            return false;
        }
        return r == 0;
#endif
    }

    int exitCode() const { return m_exitCode; }
    long pid() const {
#if defined(_WIN32)
        return static_cast<long>(m_pid);
#else
        return static_cast<long>(m_pid);
#endif
    }

    // Graceful stop: SIGINT (POSIX) / synthetic Insert (Windows).
    void requestStop() {
        if (!running()) return;
#if defined(_WIN32)
        sendInsert(true);
        m_releaseInsertAt = GetTickCount64() + 150; // hold long enough to be seen
        m_insertHeld = true;
#else
        if (m_pid > 0) {
            ::kill(m_pid, SIGINT);
        }
#endif
    }

    // Graceful stop followed by a hard kill if the child does not exit in time.
    // Returns true if the child is gone afterwards.
    bool stopAndWait(int timeoutMs) {
        if (!running()) return true;
        requestStop();
        const int stepMs = 50;
        int waited = 0;
        while (waited < timeoutMs) {
#if defined(_WIN32)
            Sleep(stepMs);
#else
            usleep(stepMs * 1000);
#endif
            waited += stepMs;
            if (!running()) return true;
        }
        return kill();
    }

    // Hard kill. Returns true if the child is gone afterwards.
    bool kill() {
        if (!running()) return true;
#if defined(_WIN32)
        if (m_proc != nullptr) {
            TerminateProcess(m_proc, 1);
            WaitForSingleObject(m_proc, 5000);
            DWORD code = 0;
            GetExitCodeProcess(m_proc, &code);
            m_exitCode = static_cast<int>(code);
            CloseHandle(m_proc);
            m_proc = nullptr;
        }
        if (m_job) { CloseHandle(m_job); m_job = nullptr; }
        return true;
#else
        if (m_pid > 0) {
            ::kill(m_pid, SIGKILL);
            int status = 0;
            (void)waitpid(m_pid, &status, 0); // reap
            m_exitCode = -1;
            m_pid = -1;
        }
        return true;
#endif
    }

private:
    void reset() {
        m_exitCode = -99;
#if defined(_WIN32)
        m_insertHeld = false;
        m_releaseInsertAt = 0;
#endif
    }

    void closeHandles() {
#if defined(_WIN32)
        if (m_proc) { CloseHandle(m_proc); m_proc = nullptr; }
        if (m_job) { CloseHandle(m_job); m_job = nullptr; }
#endif
    }

#if defined(_WIN32)
    static std::wstring widen(const std::string &s) {
        if (s.empty()) return std::wstring();
        int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()),
                                    nullptr, 0);
        std::wstring out(static_cast<size_t>(n), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()),
                            out.data(), n);
        return out;
    }

    static void sendInsert(bool down) {
        INPUT in{};
        in.type = INPUT_KEYBOARD;
        in.ki.wVk = VK_INSERT;
        in.ki.dwFlags = down ? 0 : KEYEVENTF_KEYUP;
        SendInput(1, &in, sizeof(in));
    }

    void finishWindowsStopIfDue() {
        if (m_insertHeld && GetTickCount64() >= m_releaseInsertAt) {
            sendInsert(false);
            m_insertHeld = false;
        }
    }

    void finishWindowsStop() {
        if (m_insertHeld) {
            sendInsert(false);
            m_insertHeld = false;
        }
    }

    HANDLE m_proc = nullptr;
    HANDLE m_job = nullptr;
    DWORD m_pid = 0;
    bool m_insertHeld = false;
    ULONGLONG m_releaseInsertAt = 0;
#else
    pid_t m_pid = -1;
#endif
    int m_exitCode = -99;
};

} // namespace tui
