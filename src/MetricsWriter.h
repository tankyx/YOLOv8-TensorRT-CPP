#pragma once
// MetricsWriter — writes a compact JSON status file once per second for
// external monitoring (e.g. codewhale read_file polling). Overwrites the
// same file every write; safe on NTFS for files under 4 KB.
//
// Format (status.json):
// {
//   "ts": 1717939200,
//   "fps": 238,
//   "detections": 1.8,
//   "capture": {"avg": 2.1, "min": 1.8, "max": 4.3},
//   "detect":  {"avg": 0.09, "min": 0.07, "max": 0.15},
//   "render":  {"avg": 0.5, "min": 0.3, "max": 1.2},
//   "model": "v8",
//   "graph": true,
//   "precision": "fp16",
//   "uptime_s": 3421,
//   "fw": 640, "fh": 640,
//   "boxes": [[x,y,w,h,conf,label], ...]  // up to 8, capture-frame pixels
// }

#include <chrono>
#include <fstream>
#include <string>
#include <cstdio>
#include <vector>

// Latest per-frame detection boxes, capture-frame pixels (origin = crop
// top-left). Frame centre == crosshair. Published for external monitors.
struct MetricsBox {
    float x, y, w, h, conf;
    int label;
};

class MetricsWriter {
public:
    MetricsWriter() = default;

    void open(const std::string &path) {
        m_path = path;
        m_startTime = std::chrono::steady_clock::now();
    }

    // Call from the main loop once per second with current stats.
    // Writes nothing if an optional path was never configured.
    void update(double capAvg, double capMin, double capMax,
                double detAvg, double detMin, double detMax,
                double renAvg, double renMin, double renMax,
                double detectionsPerFrame,
                const char *modelVersion,
                bool graphCaptured,
                const char *precision,
                const std::vector<MetricsBox> &boxes = {},
                int frameW = 0,
                int frameH = 0) {
        if (m_path.empty()) return;

        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - m_startTime).count();
        const auto ts = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();

        // Write to a temp file then rename for atomic replacement.
        const std::string tmpPath = m_path + ".tmp";

        // Fixed payload first; the box array is appended only while it fits, so
        // a long box list can never lose the base stats or the whole write.
        char buf[2048];
        int n = snprintf(buf, sizeof(buf),
            "{\"ts\":%lld,\"capture\":{\"avg\":%.2f,\"min\":%.2f,\"max\":%.2f},"
            "\"detect\":{\"avg\":%.2f,\"min\":%.2f,\"max\":%.2f},"
            "\"render\":{\"avg\":%.2f,\"min\":%.2f,\"max\":%.2f},"
            "\"detections\":%.1f,\"model\":\"%s\",\"graph\":%s,\"precision\":\"%s\","
            "\"uptime_s\":%.0f",
            static_cast<long long>(ts),
            capAvg, capMin, capMax,
            detAvg, detMin, detMax,
            renAvg, renMin, renMax,
            detectionsPerFrame,
            modelVersion,
            graphCaptured ? "true" : "false",
            precision,
            elapsed);

        if (n <= 0 || static_cast<size_t>(n) >= sizeof(buf)) return;

        // Box snapshot: newest frame only, capture-frame pixels. Capped at 8 so
        // the file stays small enough for tail reads.
        if (!boxes.empty()) {
            const int m = snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n),
                                   ",\"fw\":%d,\"fh\":%d,\"boxes\":[",
                                   frameW, frameH);
            if (m > 0 && static_cast<size_t>(n + m) < sizeof(buf)) n += m;
            const size_t maxBoxes = boxes.size() < 8 ? boxes.size() : 8;
            for (size_t i = 0; i < maxBoxes; ++i) {
                const MetricsBox &b = boxes[i];
                const int k = snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n),
                                       "%s[%.1f,%.1f,%.1f,%.1f,%.2f,%d]",
                                       i == 0 ? "" : ",",
                                       b.x, b.y, b.w, b.h, b.conf, b.label);
                if (k <= 0 || static_cast<size_t>(n + k) >= sizeof(buf)) break;
                n += k;
            }
            const int c = snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n), "]}");
            if (c > 0 && static_cast<size_t>(n + c) < sizeof(buf)) n += c;
        } else {
            const int c = snprintf(buf + n, sizeof(buf) - static_cast<size_t>(n), "}");
            if (c > 0 && static_cast<size_t>(n + c) < sizeof(buf)) n += c;
        }
        if (static_cast<size_t>(n) + 1 < sizeof(buf)) { buf[n++] = '\n'; buf[n] = '\0'; }

        // Write to tmp, then rename. On NTFS rename is atomic for small files.
        {
            std::ofstream ofs(tmpPath, std::ios::trunc);
            if (!ofs) return;
            ofs.write(buf, n);
            ofs.close();
        }
        std::rename(tmpPath.c_str(), m_path.c_str());
    }

private:
    std::string m_path;
    std::chrono::steady_clock::time_point m_startTime;
};
