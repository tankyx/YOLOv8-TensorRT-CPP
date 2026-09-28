// LinuxCapture.cpp
//
// Wayland screen capture via xdg-desktop-portal ScreenCast + PipeWire.
//
// Flow:
//   1. D-Bus (session bus, libdbus-1): org.freedesktop.portal.ScreenCast
//      CreateSession -> SelectSources (monitor) -> Start (KDE consent dialog on first run)
//      -> OpenPipeWireRemote (returns a PipeWire fd).
//   2. PipeWire: pw_thread_loop + pw_context + pw_core (pw_context_connect_fd on the portal
//      fd) + pw_stream bound to the portal-provided node id. Buffers are negotiated as raw
//      video, BGRA/BGRx preferred (RGBA/RGBx accepted with a byte swizzle on copy).
//   3. The stream's process callback copies the newest buffer into a CUDA pinned host buffer
//      under a mutex and bumps a frame sequence number.
//   4. CaptureScreen() stages the pinned buffer (double-buffered, so an in-flight async H2D
//      copy from the previous iteration is never overwritten) and enqueues cudaMemcpy2DAsync
//      into the caller's GpuMat.
//
// Dependencies: libpipewire-0.3, libdbus-1, CUDA runtime, OpenCV core/cuda.

#include "LinuxCapture.h"

#include <dbus/dbus.h>
#include <pipewire/pipewire.h>
#include <spa/param/buffers.h>
#include <spa/param/video/format-utils.h>
#include <spa/pod/builder.h>
#include <spa/utils/result.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

// ---------------------------------------------------------------------------
// D-Bus helpers
// ---------------------------------------------------------------------------

constexpr const char *kPortalBusName = "org.freedesktop.portal.Desktop";
constexpr const char *kPortalObjectPath = "/org/freedesktop/portal/desktop";
constexpr const char *kPortalScreenCastIface = "org.freedesktop.portal.ScreenCast";
constexpr const char *kPortalRequestIface = "org.freedesktop.portal.Request";

// Append one a{sv} entry with a string ('s') value.
bool dictAppendString(DBusMessageIter *dict, const char *key, const char *value) {
    DBusMessageIter entry, variant;
    if (!dbus_message_iter_open_container(dict, DBUS_TYPE_DICT_ENTRY, nullptr, &entry)) {
        return false;
    }
    dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
    dbus_message_iter_open_container(&entry, DBUS_TYPE_VARIANT, DBUS_TYPE_STRING_AS_STRING, &variant);
    dbus_message_iter_append_basic(&variant, DBUS_TYPE_STRING, &value);
    dbus_message_iter_close_container(&entry, &variant);
    dbus_message_iter_close_container(dict, &entry);
    return true;
}

// Append one a{sv} entry with a uint32 ('u') value.
bool dictAppendUInt32(DBusMessageIter *dict, const char *key, uint32_t value) {
    DBusMessageIter entry, variant;
    if (!dbus_message_iter_open_container(dict, DBUS_TYPE_DICT_ENTRY, nullptr, &entry)) {
        return false;
    }
    dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
    dbus_message_iter_open_container(&entry, DBUS_TYPE_VARIANT, DBUS_TYPE_UINT32_AS_STRING, &variant);
    dbus_message_iter_append_basic(&variant, DBUS_TYPE_UINT32, &value);
    dbus_message_iter_close_container(&entry, &variant);
    dbus_message_iter_close_container(dict, &entry);
    return true;
}

// Append one a{sv} entry with a boolean ('b') value.
bool dictAppendBool(DBusMessageIter *dict, const char *key, bool value) {
    DBusMessageIter entry, variant;
    dbus_bool_t v = value ? TRUE : FALSE;
    if (!dbus_message_iter_open_container(dict, DBUS_TYPE_DICT_ENTRY, nullptr, &entry)) {
        return false;
    }
    dbus_message_iter_append_basic(&entry, DBUS_TYPE_STRING, &key);
    dbus_message_iter_open_container(&entry, DBUS_TYPE_VARIANT, DBUS_TYPE_BOOLEAN_AS_STRING, &variant);
    dbus_message_iter_append_basic(&variant, DBUS_TYPE_BOOLEAN, &v);
    dbus_message_iter_close_container(&entry, &variant);
    dbus_message_iter_close_container(dict, &entry);
    return true;
}

// Unique tokens so several portal requests/sessions never collide.
std::string makeToken(const char *prefix) {
    static std::atomic<unsigned long> counter{0};
    return std::string(prefix) + "_" + std::to_string(::getpid()) + "_" + std::to_string(counter.fetch_add(1));
}

// The portal builds request/session object paths from our unique bus name:
// ":1.234" -> "1_234".
std::string senderToken(DBusConnection *conn) {
    const char *unique = dbus_bus_get_unique_name(conn);
    std::string s = unique ? unique : "";
    if (!s.empty() && s[0] == ':') {
        s.erase(0, 1);
    }
    for (char &c : s) {
        if (c == '.') {
            c = '_';
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Restore token (screen-cast permission memory)
// ---------------------------------------------------------------------------
// A portal backend that accepts persist_mode=2 issues a `restore_token` in the Start reply.
// The token is the handle to the permission the user already granted, so passing it back in
// SelectSources is what makes later runs skip the consent dialog. Without it every launch
// looks like a first-time request and the dialog pops up again.
// Per-user, per-machine state -> $XDG_STATE_HOME (default ~/.local/state), not the checkout.
std::filesystem::path restoreTokenPath() {
    const char *overridePath = std::getenv("YOLO_SCREENCAST_TOKEN_FILE");
    if (overridePath && *overridePath) {
        return std::filesystem::path(overridePath);
    }
    const char *stateHome = std::getenv("XDG_STATE_HOME");
    std::filesystem::path base;
    if (stateHome && *stateHome) {
        base = stateHome;
    } else {
        const char *home = std::getenv("HOME");
        base = (home && *home) ? std::filesystem::path(home) / ".local" / "state"
                               : std::filesystem::temp_directory_path();
    }
    return base / "yolov8-tensorrt-cpp" / "screencast-restore-token";
}

// Returns the remembered token, or an empty string when there is none to reuse.
std::string loadRestoreToken() {
    std::ifstream ifs(restoreTokenPath());
    std::string token;
    if (!ifs) {
        return token;
    }
    std::getline(ifs, token);
    const auto notSpace = [](unsigned char c) { return std::isspace(c) == 0; };
    token.erase(token.begin(), std::find_if(token.begin(), token.end(), notSpace));
    token.erase(std::find_if(token.rbegin(), token.rend(), notSpace).base(), token.end());
    return token;
}

// Best effort: a token we fail to store only costs one extra consent dialog next run.
void saveRestoreToken(const std::string &token) {
    const std::filesystem::path path = restoreTokenPath();
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    const std::string tmpPath = path.string() + ".tmp";
    // 0600: the token stands in for a granted permission, treat it as a secret.
    const int fd = ::open(tmpPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
        std::cerr << "LinuxCapture: cannot write " << tmpPath << ": " << std::strerror(errno)
                  << std::endl;
        return;
    }
    const ssize_t written = ::write(fd, token.data(), token.size());
    ::close(fd);
    if (written != static_cast<ssize_t>(token.size())) {
        std::cerr << "LinuxCapture: short write of " << tmpPath << std::endl;
        ::unlink(tmpPath.c_str());
        return;
    }
    // Atomic replacement, same pattern as MetricsWriter.
    if (std::rename(tmpPath.c_str(), path.c_str()) != 0) {
        std::cerr << "LinuxCapture: cannot rename " << tmpPath << " to " << path.string() << ": "
                  << std::strerror(errno) << std::endl;
        ::unlink(tmpPath.c_str());
    }
}

void addMatch(DBusConnection *conn, const std::string &rule) {
    DBusError err;
    dbus_error_init(&err);
    dbus_bus_add_match(conn, rule.c_str(), &err);
    dbus_connection_flush(conn);
    if (dbus_error_is_set(&err)) {
        std::cerr << "LinuxCapture: dbus_bus_add_match failed: " << err.message << std::endl;
        dbus_error_free(&err);
    }
}

void removeMatch(DBusConnection *conn, const std::string &rule) {
    DBusError err;
    dbus_error_init(&err);
    dbus_bus_remove_match(conn, rule.c_str(), &err);
    dbus_connection_flush(conn);
    if (dbus_error_is_set(&err)) {
        dbus_error_free(&err);
    }
}

// Wait for the org.freedesktop.portal.Request.Response signal on `requestPath`.
// On success returns true, fills `responseCode` (0 == success) and, when the response is a
// success and `parseResults` is set, hands the a{sv} results iterator to the callback.
// Returns false on timeout.
bool waitForResponse(DBusConnection *conn, const std::string &requestPath, int timeoutMs,
                     uint32_t &responseCode, const std::function<void(DBusMessageIter *)> &parseResults) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        dbus_connection_read_write(conn, 100);
        while (DBusMessage *msg = dbus_connection_pop_message(conn)) {
            const bool match =
                dbus_message_is_signal(msg, kPortalRequestIface, "Response") &&
                dbus_message_get_path(msg) != nullptr && requestPath == dbus_message_get_path(msg);
            if (match) {
                DBusMessageIter args;
                if (!dbus_message_iter_init(msg, &args) ||
                    dbus_message_iter_get_arg_type(&args) != DBUS_TYPE_UINT32) {
                    dbus_message_unref(msg);
                    continue;
                }
                dbus_message_iter_get_basic(&args, &responseCode);
                if (responseCode == 0 && parseResults && dbus_message_iter_next(&args) &&
                    dbus_message_iter_get_arg_type(&args) == DBUS_TYPE_ARRAY) {
                    parseResults(&args);
                }
                dbus_message_unref(msg);
                return true;
            }
            dbus_message_unref(msg);
        }
    }
    return false;
}

// Iterate a portal a{sv} results dict, invoking `visitor(key, variantIter)` per entry.
void forEachResultEntry(DBusMessageIter *results, const std::function<void(const char *, DBusMessageIter *)> &visitor) {
    DBusMessageIter dict;
    dbus_message_iter_recurse(results, &dict);
    while (dbus_message_iter_get_arg_type(&dict) == DBUS_TYPE_DICT_ENTRY) {
        DBusMessageIter entry;
        dbus_message_iter_recurse(&dict, &entry);
        if (dbus_message_iter_get_arg_type(&entry) == DBUS_TYPE_STRING) {
            const char *key = nullptr;
            dbus_message_iter_get_basic(&entry, &key);
            if (key && dbus_message_iter_next(&entry) &&
                dbus_message_iter_get_arg_type(&entry) == DBUS_TYPE_VARIANT) {
                DBusMessageIter variant;
                dbus_message_iter_recurse(&entry, &variant);
                visitor(key, &variant);
            }
        }
        dbus_message_iter_next(&dict);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// LinuxCapture::Impl
// ---------------------------------------------------------------------------

struct LinuxCapture::Impl {
    Impl() {
        try {
            portalHandshake();
            setupPipeWire();
        } catch (...) {
            teardown();
            throw;
        }
    }

    ~Impl() { teardown(); }

    // ---- Portal (D-Bus) -------------------------------------------------

    DBusConnection *dbus = nullptr;
    std::string sessionPath;
    uint32_t pwNodeId = 0;

    void portalHandshake() {
        DBusError err;
        dbus_error_init(&err);
        dbus = dbus_bus_get_private(DBUS_BUS_SESSION, &err);
        if (!dbus) {
            std::string msg = err.message ? err.message : "unknown error";
            dbus_error_free(&err);
            throw std::runtime_error("LinuxCapture: cannot connect to the session D-Bus: " + msg);
        }
        dbus_connection_set_exit_on_disconnect(dbus, FALSE);

        const std::string sender = senderToken(dbus);

        // -- CreateSession -------------------------------------------------
        const std::string sessionToken = makeToken("lc_session");
        sessionPath = "/org/freedesktop/portal/desktop/session/" + sender + "/" + sessionToken;
        portalCall("CreateSession", nullptr, [&](DBusMessageIter *dict) {
            dictAppendString(dict, "session_handle_token", sessionToken.c_str());
        }, [&](DBusMessageIter *results) {
            // Prefer the session handle reported in the Response results.
            forEachResultEntry(results, [&](const char *key, DBusMessageIter *variant) {
                if (std::strcmp(key, "session_handle") == 0) {
                    const int t = dbus_message_iter_get_arg_type(variant);
                    if (t == DBUS_TYPE_OBJECT_PATH || t == DBUS_TYPE_STRING) {
                        const char *p = nullptr;
                        dbus_message_iter_get_basic(variant, &p);
                        if (p && *p) {
                            sessionPath = p;
                        }
                    }
                }
            });
        });

        // -- SelectSources --------------------------------------------------
        // persist_mode=2 (persist until revoked) plus the restore token remembered from an
        // earlier run are what let the portal skip its consent dialog: the token is the handle
        // to the permission the user already granted. Neither is essential to capture, so drop
        // the optional arguments step by step instead of failing the whole session.
        const std::string storedRestoreToken = loadRestoreToken();
        auto selectSources = [&](bool withPersist, bool withToken) {
            portalCall("SelectSources", [&](DBusMessageIter *args) {
                const char *sp = sessionPath.c_str();
                dbus_message_iter_append_basic(args, DBUS_TYPE_OBJECT_PATH, &sp);
            }, [&](DBusMessageIter *dict) {
                dictAppendUInt32(dict, "types", 1); // 1 = monitor
                dictAppendBool(dict, "multiple", false);
                if (withPersist) {
                    dictAppendUInt32(dict, "persist_mode", 2);
                }
                if (withToken && !storedRestoreToken.empty()) {
                    dictAppendString(dict, "restore_token", storedRestoreToken.c_str());
                }
            }, nullptr);
        };
        const struct {
            bool persist;
            bool token;
            const char *label;
        } selectVariants[] = {
            {true, true, "persist_mode=2 + restore_token"},
            {true, false, "persist_mode=2"},
            {false, false, "no persistence"},
        };
        bool sourcesSelected = false;
        std::string selectError;
        for (const auto &variant : selectVariants) {
            try {
                selectSources(variant.persist, variant.token);
                sourcesSelected = true;
                break;
            } catch (const std::exception &e) {
                selectError = e.what();
                std::cerr << "LinuxCapture: SelectSources (" << variant.label << ") failed: "
                          << selectError << std::endl;
            }
        }
        if (!sourcesSelected) {
            throw std::runtime_error("LinuxCapture: portal SelectSources failed: " + selectError);
        }

        // -- Start ----------------------------------------------------------
        // Empty parent_window: KDE shows its own dialog. This is where the user consents.
        portalCall("Start", [&](DBusMessageIter *args) {
            const char *sp = sessionPath.c_str();
            dbus_message_iter_append_basic(args, DBUS_TYPE_OBJECT_PATH, &sp);
            const char *parentWindow = "";
            dbus_message_iter_append_basic(args, DBUS_TYPE_STRING, &parentWindow);
        }, nullptr, [&](DBusMessageIter *results) {
            // The token that lets the next run skip this dialog; it is what persist_mode=2
            // actually buys us. Some backends omit it, in which case we keep the old one.
            forEachResultEntry(results, [&](const char *key, DBusMessageIter *variant) {
                if (std::strcmp(key, "restore_token") == 0 &&
                    dbus_message_iter_get_arg_type(variant) == DBUS_TYPE_STRING) {
                    const char *token = nullptr;
                    dbus_message_iter_get_basic(variant, &token);
                    if (token && *token) {
                        saveRestoreToken(token);
                        std::cout << "LinuxCapture: remembered screen-cast permission; later runs "
                                     "skip the consent dialog."
                                  << std::endl;
                    }
                    return;
                }
                // results["streams"] : a(ua{sv}) — first struct's u is the PipeWire node id.
                if (std::strcmp(key, "streams") != 0 ||
                    dbus_message_iter_get_arg_type(variant) != DBUS_TYPE_ARRAY || pwNodeId != 0) {
                    return;
                }
                DBusMessageIter streams;
                dbus_message_iter_recurse(variant, &streams);
                while (dbus_message_iter_get_arg_type(&streams) == DBUS_TYPE_STRUCT) {
                    DBusMessageIter streamEntry;
                    dbus_message_iter_recurse(&streams, &streamEntry);
                    if (dbus_message_iter_get_arg_type(&streamEntry) == DBUS_TYPE_UINT32) {
                        dbus_message_iter_get_basic(&streamEntry, &pwNodeId);
                    }
                    break; // only one stream requested (multiple=false)
                }
            });
        });
        if (pwNodeId == 0) {
            throw std::runtime_error("LinuxCapture: portal Start response contained no PipeWire streams.");
        }

        // -- OpenPipeWireRemote ---------------------------------------------
        // Returns a connected PipeWire remote fd ('h') for this session.
        DBusMessage *msg = dbus_message_new_method_call(kPortalBusName, kPortalObjectPath,
                                                        kPortalScreenCastIface, "OpenPipeWireRemote");
        if (!msg) {
            throw std::runtime_error("LinuxCapture: out of memory building D-Bus message.");
        }
        {
            DBusMessageIter args;
            dbus_message_iter_init_append(msg, &args);
            const char *sp = sessionPath.c_str();
            dbus_message_iter_append_basic(&args, DBUS_TYPE_OBJECT_PATH, &sp);
            DBusMessageIter dict;
            dbus_message_iter_open_container(&args, DBUS_TYPE_ARRAY, "{sv}", &dict);
            dbus_message_iter_close_container(&args, &dict);
        }
        DBusError err2;
        dbus_error_init(&err2);
        DBusMessage *reply = dbus_connection_send_with_reply_and_block(dbus, msg, 5000, &err2);
        dbus_message_unref(msg);
        if (!reply) {
            std::string e = err2.message ? err2.message : "unknown D-Bus error";
            dbus_error_free(&err2);
            throw std::runtime_error("LinuxCapture: OpenPipeWireRemote failed: " + e);
        }
        DBusMessageIter replyArgs;
        if (!dbus_message_iter_init(reply, &replyArgs) ||
            dbus_message_iter_get_arg_type(&replyArgs) != DBUS_TYPE_UNIX_FD) {
            dbus_message_unref(reply);
            throw std::runtime_error("LinuxCapture: OpenPipeWireRemote reply did not carry a unix fd "
                                     "(portal too old or fd passing unavailable).");
        }
        dbus_message_iter_get_basic(&replyArgs, &portalFd);
        // Dup right away: the fd owned by the message dies with the unref below.
        portalFd = ::fcntl(portalFd, F_DUPFD_CLOEXEC, 3);
        dbus_message_unref(reply);
        if (portalFd < 0) {
            throw std::runtime_error("LinuxCapture: failed to duplicate the portal PipeWire fd.");
        }
    }

    // One portal method call with a predictable request handle: a fresh handle_token is
    // generated here, the Request.Response match is installed *before* the call goes out,
    // and handle_token is injected into the trailing a{sv} options dict of every call.
    //   appendArgs   — appends the method arguments that precede the options dict (may be null)
    //   fillOptions  — appends extra entries into the options dict (may be null)
    //   parseResults — receives the a{sv} results of a successful Response (may be null)
    // Throws std::runtime_error on D-Bus error, user refusal, failure, or timeout.
    void portalCall(const char *method, const std::function<void(DBusMessageIter *)> &appendArgs,
                    const std::function<void(DBusMessageIter *)> &fillOptions,
                    const std::function<void(DBusMessageIter *)> &parseResults) {
        const std::string token = makeToken("lc_req");
        const std::string requestPath =
            "/org/freedesktop/portal/desktop/request/" + senderToken(dbus) + "/" + token;
        const std::string matchRule = std::string("type='signal',interface='") + kPortalRequestIface +
                                      "',path='" + requestPath + "'";
        addMatch(dbus, matchRule);

        try {
            DBusMessage *msg = dbus_message_new_method_call(kPortalBusName, kPortalObjectPath,
                                                            kPortalScreenCastIface, method);
            if (!msg) {
                throw std::runtime_error("out of memory building D-Bus message");
            }

            DBusMessageIter args;
            dbus_message_iter_init_append(msg, &args);
            if (appendArgs) {
                appendArgs(&args);
            }
            DBusMessageIter dict;
            dbus_message_iter_open_container(&args, DBUS_TYPE_ARRAY, "{sv}", &dict);
            if (fillOptions) {
                fillOptions(&dict);
            }
            dictAppendString(&dict, "handle_token", token.c_str());
            dbus_message_iter_close_container(&args, &dict);

            DBusError err;
            dbus_error_init(&err);
            DBusMessage *reply = dbus_connection_send_with_reply_and_block(dbus, msg, 5000, &err);
            dbus_message_unref(msg);
            if (!reply) {
                std::string e = err.message ? err.message : "unknown D-Bus error";
                dbus_error_free(&err);
                throw std::runtime_error(std::string("portal call ") + method + " failed: " + e);
            }
            if (dbus_message_get_type(reply) == DBUS_MESSAGE_TYPE_ERROR) {
                const char *e = dbus_message_get_error_name(reply);
                std::string es = e ? e : "unknown portal error";
                dbus_message_unref(reply);
                throw std::runtime_error(std::string("portal call ") + method + " failed: " + es);
            }
            dbus_message_unref(reply);

            uint32_t responseCode = 2;
            // Start pops the KDE dialog: allow generous time. Others are instant.
            const int timeoutMs = std::strcmp(method, "Start") == 0 ? 180000 : 15000;
            if (!waitForResponse(dbus, requestPath, timeoutMs, responseCode, parseResults)) {
                throw std::runtime_error(std::string("timed out waiting for portal Response to ") + method);
            }
            if (responseCode == 1) {
                throw std::runtime_error(std::string("portal request ") + method + " cancelled by the user");
            }
            if (responseCode != 0) {
                throw std::runtime_error(std::string("portal request ") + method + " failed (response " +
                                         std::to_string(responseCode) + ")");
            }
        } catch (...) {
            removeMatch(dbus, matchRule);
            throw;
        }
        removeMatch(dbus, matchRule);
    }

    // ---- PipeWire --------------------------------------------------------

    pw_thread_loop *loop = nullptr;
    pw_context *context = nullptr;
    pw_core *core = nullptr;
    pw_stream *stream = nullptr;
    spa_hook streamListener{};
    // Must outlive the stream: pw_stream_add_listener stores a POINTER to this
    // struct (listener->funcs = events), it does not copy it. A stack-local
    // pw_stream_events dangles as soon as setupPipeWire() returns and the
    // add_buffer/state_changed emits jump through garbage — this was the
    // pw_impl_port_use_buffers SIGSEGV.
    pw_stream_events streamEvents{};
    int portalFd = -1;

    std::string pwError; // set from state_changed on the loop thread
    pw_stream_state lastState = PW_STREAM_STATE_UNCONNECTED;

    // Latest negotiated format + newest frame (process callback -> CaptureScreen).
    std::mutex frameMutex;
    int width = 0;              // negotiated frame size (set by onParamChanged)
    int height = 0;
    size_t negotiatedBytes = 0; // width*height*4 of the negotiated format
    bool swapRB = false;        // true when the negotiated format is RGBA/RGBx instead of BGRA/BGRx
    uint8_t *pinned = nullptr;  // cudaHostAlloc, allocated on the caller thread, written by the PipeWire thread
    size_t pinnedBytes = 0;     // 0 until `pinned` matches the negotiated format
    uint64_t frameSeq = 0;

    // CaptureScreen side.
    uint8_t *staging[2] = {nullptr, nullptr};
    size_t stagingBytes = 0;
    int stagingIndex = 0;
    uint64_t lastSeq = 0;

    static void pwInitOnce() {
        static std::once_flag flag;
        std::call_once(flag, [] { pw_init(nullptr, nullptr); });
    }

    void setupPipeWire() {
        pwInitOnce();

        // Duplicate the portal fd: pw_context_connect_fd takes ownership of the fd it gets.
        int fd = ::fcntl(portalFd, F_DUPFD_CLOEXEC, 3);
        if (fd < 0) {
            throw std::runtime_error("LinuxCapture: failed to duplicate the portal PipeWire fd.");
        }

        loop = pw_thread_loop_new("linux-capture", nullptr);
        if (!loop) {
            ::close(fd);
            throw std::runtime_error("LinuxCapture: pw_thread_loop_new failed.");
        }
        context = pw_context_new(pw_thread_loop_get_loop(loop), nullptr, 0);
        if (!context) {
            ::close(fd);
            throw std::runtime_error("LinuxCapture: pw_context_new failed.");
        }
        if (pw_thread_loop_start(loop) < 0) {
            ::close(fd);
            throw std::runtime_error("LinuxCapture: pw_thread_loop_start failed.");
        }

        pw_thread_loop_lock(loop);

        core = pw_context_connect_fd(context, fd, nullptr, 0);
        if (!core) {
            pw_thread_loop_unlock(loop);
            ::close(fd);
            throw std::runtime_error("LinuxCapture: pw_context_connect_fd failed "
                                     "(portal fd rejected by PipeWire).");
        }

        pw_properties *props = pw_properties_new(PW_KEY_MEDIA_TYPE, "Video",
                                                 PW_KEY_MEDIA_CATEGORY, "Capture",
                                                 PW_KEY_MEDIA_ROLE, "Screen", nullptr);
        stream = pw_stream_new(core, "linux-capture-stream", props);
        if (!stream) {
            pw_thread_loop_unlock(loop);
            throw std::runtime_error("LinuxCapture: pw_stream_new failed.");
        }

        streamEvents.version = PW_VERSION_STREAM_EVENTS;
        streamEvents.state_changed = &Impl::onStateChanged;
        streamEvents.param_changed = &Impl::onParamChanged;
        streamEvents.process = &Impl::onProcess;
        pw_stream_add_listener(stream, &streamListener, &streamEvents, this);

        // Offer raw video, BGRA/BGRx preferred with RGBA/RGBx fallback.
        // (Named locals: C++ forbids taking the address of the SPA compound literals.)
        const spa_rectangle defSize = SPA_RECTANGLE(1920, 1080);
        const spa_rectangle minSize = SPA_RECTANGLE(1, 1);
        const spa_rectangle maxSize = SPA_RECTANGLE(8192, 8192);
        const spa_fraction defFps = SPA_FRACTION(0, 1);
        const spa_fraction minFps = SPA_FRACTION(0, 1);
        const spa_fraction maxFps = SPA_FRACTION(1000, 1);
        uint8_t paramsBuf[1024];
        spa_pod_builder builder = SPA_POD_BUILDER_INIT(paramsBuf, sizeof(paramsBuf));
        const spa_pod *params[1];
        params[0] = static_cast<const spa_pod *>(spa_pod_builder_add_object(
            &builder, SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
            SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
            SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
            SPA_FORMAT_VIDEO_format,
            SPA_POD_CHOICE_ENUM_Id(5, SPA_VIDEO_FORMAT_BGRA, SPA_VIDEO_FORMAT_BGRA, SPA_VIDEO_FORMAT_BGRx,
                                   SPA_VIDEO_FORMAT_RGBA, SPA_VIDEO_FORMAT_RGBx),
            SPA_FORMAT_VIDEO_size, SPA_POD_CHOICE_RANGE_Rectangle(&defSize, &minSize, &maxSize),
            SPA_FORMAT_VIDEO_framerate, SPA_POD_CHOICE_RANGE_Fraction(&defFps, &minFps, &maxFps)));

        int ret = pw_stream_connect(stream, PW_DIRECTION_INPUT, pwNodeId,
                                    static_cast<pw_stream_flags>(PW_STREAM_FLAG_AUTOCONNECT |
                                                                 PW_STREAM_FLAG_MAP_BUFFERS),
                                    params, 1);
        if (ret < 0) {
            pw_thread_loop_unlock(loop);
            throw std::runtime_error(std::string("LinuxCapture: pw_stream_connect failed: ") +
                                     spa_strerror(ret));
        }

        // Wait (bounded) until the stream is up AND the format is negotiated —
        // screenWidth()/screenHeight() are invalid (0) until param_changed ran,
        // and callers use them for geometry right after construction.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        while (true) {
            const char *err = nullptr;
            lastState = pw_stream_get_state(stream, &err);
            bool formatReady = false;
            {
                std::lock_guard<std::mutex> lk(frameMutex);
                formatReady = width > 0 && height > 0;
            }
            if ((lastState == PW_STREAM_STATE_STREAMING || lastState == PW_STREAM_STATE_PAUSED) &&
                formatReady) {
                break;
            }
            if (lastState == PW_STREAM_STATE_ERROR) {
                std::string e = err ? err : "unknown PipeWire stream error";
                pw_thread_loop_unlock(loop);
                throw std::runtime_error("LinuxCapture: PipeWire stream error: " + e);
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                pw_thread_loop_unlock(loop);
                throw std::runtime_error("LinuxCapture: timed out waiting for the PipeWire stream "
                                         "format negotiation.");
            }
            pw_thread_loop_timed_wait(loop, 1);
        }
        pw_thread_loop_unlock(loop);
    }

    // Runs on the PipeWire loop thread.
    static void onStateChanged(void *userdata, pw_stream_state oldState, pw_stream_state newState,
                               const char *error) {
        Impl *self = static_cast<Impl *>(userdata);
        (void)oldState;
        self->lastState = newState;
        if (newState == PW_STREAM_STATE_ERROR && error) {
            self->pwError = error;
            std::cerr << "LinuxCapture: PipeWire stream error: " << error << std::endl;
        }
        pw_thread_loop_signal(self->loop, false);
    }

    // Runs on the PipeWire loop thread: format negotiated (or re-negotiated mid-stream).
    static void onParamChanged(void *userdata, uint32_t id, const spa_pod *param) {
        Impl *self = static_cast<Impl *>(userdata);
        if (id != SPA_PARAM_Format || param == nullptr) {
            return;
        }

        uint32_t mediaType = 0, mediaSubtype = 0;
        if (spa_format_parse(param, &mediaType, &mediaSubtype) < 0 ||
            mediaType != SPA_MEDIA_TYPE_video || mediaSubtype != SPA_MEDIA_SUBTYPE_raw) {
            return;
        }

        spa_video_info_raw info;
        spa_zero(info);
        if (spa_format_video_raw_parse(param, &info) < 0) {
            return;
        }

        bool swap = false;
        switch (info.format) {
        case SPA_VIDEO_FORMAT_BGRA:
        case SPA_VIDEO_FORMAT_BGRx:
            swap = false;
            break;
        case SPA_VIDEO_FORMAT_RGBA:
        case SPA_VIDEO_FORMAT_RGBx:
            swap = true;
            break;
        default:
            std::cerr << "LinuxCapture: unsupported negotiated video format, ignoring." << std::endl;
            return;
        }

        const int w = static_cast<int>(info.size.width);
        const int h = static_cast<int>(info.size.height);
        if (w <= 0 || h <= 0) {
            return;
        }

        const size_t bytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 4;
        {
            std::lock_guard<std::mutex> lock(self->frameMutex);
            // Only record the negotiated format here. The pinned buffer is (re)allocated by
            // CaptureScreen on the caller's thread: CUDA driver calls (cudaHostAlloc /
            // cudaFreeHost, including first-call context setup on this thread) must never
            // run on the PipeWire realtime loop thread.
            const bool changed = w != self->width || h != self->height ||
                                 bytes != self->negotiatedBytes || swap != self->swapRB;
            self->width = w;
            self->height = h;
            self->swapRB = swap;
            self->negotiatedBytes = bytes;
            if (changed) {
                std::cerr << "LinuxCapture: negotiated " << w << "x" << h
                          << (swap ? " RGBA (swizzled to BGRA)" : " BGRA") << std::endl;
            }
        }

        // Tell the producer what buffers we accept.
        uint8_t buf[1024];
        spa_pod_builder builder = SPA_POD_BUILDER_INIT(buf, sizeof(buf));
        const int32_t stride = w * 4;
        const int32_t size = static_cast<int32_t>(bytes);
        const spa_pod *params[1];
        params[0] = static_cast<const spa_pod *>(spa_pod_builder_add_object(
            &builder, SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
            SPA_PARAM_BUFFERS_dataType,
            SPA_POD_CHOICE_FLAGS_Int((1 << SPA_DATA_MemFd) | (1 << SPA_DATA_MemPtr)),
            SPA_PARAM_BUFFERS_buffers, SPA_POD_CHOICE_RANGE_Int(4, 2, 16),
            SPA_PARAM_BUFFERS_blocks, SPA_POD_Int(1),
            SPA_PARAM_BUFFERS_size, SPA_POD_Int(size),
            SPA_PARAM_BUFFERS_stride, SPA_POD_Int(stride),
            SPA_PARAM_BUFFERS_align, SPA_POD_Int(16)));
        pw_stream_update_params(self->stream, params, 1);
    }

    // Runs on the PipeWire loop thread: copy the newest buffer into the pinned buffer.
    static void onProcess(void *userdata) {
        Impl *self = static_cast<Impl *>(userdata);

        pw_buffer *pwbuf = pw_stream_dequeue_buffer(self->stream);
        if (!pwbuf) {
            return;
        }
        spa_buffer *buf = pwbuf->buffer;
        if (!buf || buf->n_datas < 1) {
            pw_stream_queue_buffer(self->stream, pwbuf);
            return;
        }

        spa_data *data = &buf->datas[0];
        if ((data->type != SPA_DATA_MemFd && data->type != SPA_DATA_MemPtr) || data->data == nullptr) {
            pw_stream_queue_buffer(self->stream, pwbuf);
            return;
        }

        std::lock_guard<std::mutex> lock(self->frameMutex);
        // CaptureScreen (re)allocates the pinned buffer on the caller thread; until it
        // matches the negotiated format there is nowhere to copy to — drop the buffer.
        if (!self->pinned || self->pinnedBytes != self->negotiatedBytes || self->negotiatedBytes == 0) {
            pw_stream_queue_buffer(self->stream, pwbuf);
            return;
        }

        // Bound every read by the mapped size, never trust chunk->offset/size/stride blindly:
        // reading past the mmap'd spa_data region is a segfault.
        const spa_chunk *chunk = data->chunk;
        const size_t maxsize = data->maxsize;
        size_t offset = chunk ? chunk->offset : 0;
        if (offset > maxsize) {
            offset = maxsize;
        }
        size_t avail = chunk ? chunk->size : maxsize;
        if (avail > maxsize - offset) {
            avail = maxsize - offset;
        }
        const int srcStride = (chunk && chunk->stride > 0) ? chunk->stride : self->width * 4;

        const uint8_t *src = static_cast<const uint8_t *>(data->data) + offset;
        const size_t rowBytes = static_cast<size_t>(self->width) * 4;
        int rows = self->height;
        if (srcStride > 0 && avail < static_cast<size_t>(srcStride) * static_cast<size_t>(rows)) {
            rows = static_cast<int>(avail / static_cast<size_t>(srcStride));
        }
        if (rows <= 0) {
            // Producer left the buffer empty: no new frame, do not bump the sequence.
            pw_stream_queue_buffer(self->stream, pwbuf);
            return;
        }
        // Never read more than one stride per row even if the row is narrower than width*4.
        const size_t copyBytes = std::min(rowBytes, static_cast<size_t>(srcStride));

        if (!self->swapRB && srcStride == static_cast<int>(rowBytes) &&
            avail >= rowBytes * static_cast<size_t>(rows)) {
            std::memcpy(self->pinned, src, rowBytes * static_cast<size_t>(rows));
        } else {
            for (int y = 0; y < rows; ++y) {
                const uint8_t *s = src + static_cast<size_t>(y) * static_cast<size_t>(srcStride);
                uint8_t *d = self->pinned + static_cast<size_t>(y) * rowBytes;
                if (self->swapRB) {
                    const int pixels = static_cast<int>(copyBytes / 4);
                    for (int x = 0; x < pixels; ++x) {
                        d[0] = s[2];
                        d[1] = s[1];
                        d[2] = s[0];
                        d[3] = s[3];
                        s += 4;
                        d += 4;
                    }
                    if (static_cast<size_t>(pixels) * 4 < rowBytes) {
                        std::memset(d, 0, rowBytes - static_cast<size_t>(pixels) * 4);
                    }
                } else {
                    std::memcpy(d, s, copyBytes);
                    if (copyBytes < rowBytes) {
                        std::memset(d + copyBytes, 0, rowBytes - copyBytes);
                    }
                }
            }
        }
        if (rows < self->height) {
            std::memset(self->pinned + static_cast<size_t>(rows) * rowBytes, 0,
                        rowBytes * static_cast<size_t>(self->height - rows));
        }

        ++self->frameSeq;
        pw_stream_queue_buffer(self->stream, pwbuf);
    }

    // ---- CaptureScreen ---------------------------------------------------

    bool capture(cv::cuda::GpuMat &frame, cudaStream_t cudaStream) {
        std::lock_guard<std::mutex> lock(frameMutex);
        if (negotiatedBytes == 0 || width <= 0 || height <= 0) {
            return false; // stream has not negotiated a format yet
        }

        // All CUDA driver calls live on this (the caller's) thread — never on the PipeWire
        // loop thread. (Re)allocate the pinned buffer whenever the negotiated format changed;
        // the process callback simply skips buffers until pinnedBytes catches up.
        if (!pinned || pinnedBytes != negotiatedBytes) {
            if (pinned) {
                cudaFreeHost(pinned);
                pinned = nullptr;
                pinnedBytes = 0;
            }
            if (cudaHostAlloc(reinterpret_cast<void **>(&pinned), negotiatedBytes, cudaHostAllocDefault) !=
                cudaSuccess) {
                std::cerr << "LinuxCapture: cudaHostAlloc failed for " << negotiatedBytes << " bytes."
                          << std::endl;
                return false;
            }
            pinnedBytes = negotiatedBytes;
        }

        if (stagingBytes != negotiatedBytes) {
            for (auto &s : staging) {
                if (s) {
                    cudaFreeHost(s);
                    s = nullptr;
                }
            }
            stagingBytes = 0;
            if (cudaHostAlloc(reinterpret_cast<void **>(&staging[0]), negotiatedBytes, cudaHostAllocDefault) !=
                    cudaSuccess ||
                cudaHostAlloc(reinterpret_cast<void **>(&staging[1]), negotiatedBytes, cudaHostAllocDefault) !=
                    cudaSuccess) {
                std::cerr << "LinuxCapture: cudaHostAlloc failed for staging buffers." << std::endl;
                return false;
            }
            stagingBytes = negotiatedBytes;
        }

        if (frameSeq == lastSeq) {
            return false; // no new frame yet — caller retries next iteration
        }

        const size_t bytes = negotiatedBytes;

        // Stage under the mutex so the PipeWire thread cannot overwrite the frame while the
        // async H2D below is still queued. Two staging buffers alternate, giving the previous
        // iteration's copy a full extra loop iteration to complete before reuse.
        uint8_t *dst = staging[stagingIndex];
        stagingIndex ^= 1;
        std::memcpy(dst, pinned, bytes);

        if (frame.rows != height || frame.cols != width || frame.type() != CV_8UC4) {
            frame.create(height, width, CV_8UC4);
        }

        const size_t widthBytes = static_cast<size_t>(width) * 4;
        cudaError_t cerr =
            cudaMemcpy2DAsync(frame.data, frame.step, dst, widthBytes, widthBytes,
                              static_cast<size_t>(height), cudaMemcpyHostToDevice, cudaStream);
        if (cerr != cudaSuccess) {
            std::cerr << "LinuxCapture: cudaMemcpy2DAsync failed: " << cudaGetErrorString(cerr) << std::endl;
            return false;
        }

        lastSeq = frameSeq;
        return true;
    }

    // ---- Teardown ----------------------------------------------------------

    void teardown() noexcept {
        if (loop) {
            pw_thread_loop_lock(loop);
            if (stream) {
                pw_stream_destroy(stream); // also removes the listener
                stream = nullptr;
            }
            if (core) {
                pw_core_disconnect(core); // closes the fd handed to pw_context_connect_fd
                core = nullptr;
            }
            pw_thread_loop_unlock(loop);
            pw_thread_loop_stop(loop); // no callbacks run past this point
            pw_context_destroy(context);
            context = nullptr;
            pw_thread_loop_destroy(loop);
            loop = nullptr;
        }
        if (pinned) {
            cudaFreeHost(pinned);
            pinned = nullptr;
        }
        for (auto &s : staging) {
            if (s) {
                cudaFreeHost(s);
                s = nullptr;
            }
        }
        if (portalFd >= 0) {
            ::close(portalFd);
            portalFd = -1;
        }
        if (dbus) {
            dbus_connection_close(dbus);
            dbus_connection_unref(dbus);
            dbus = nullptr;
        }
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

LinuxCapture::LinuxCapture() : m_impl(std::make_unique<Impl>()) {}

LinuxCapture::~LinuxCapture() = default;

bool LinuxCapture::CaptureScreen(cv::cuda::GpuMat &frame, cudaStream_t stream) {
    return m_impl->capture(frame, stream);
}

int LinuxCapture::screenWidth() const {
    std::lock_guard<std::mutex> lock(m_impl->frameMutex);
    return m_impl->width;
}

int LinuxCapture::screenHeight() const {
    std::lock_guard<std::mutex> lock(m_impl->frameMutex);
    return m_impl->height;
}
