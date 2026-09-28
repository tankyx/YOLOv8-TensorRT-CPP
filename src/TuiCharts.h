#pragma once
// TuiCharts — pure ASCII rendering helpers for the detector TUI.
//
// Every function here is pure (numbers in, text out), so the charts can be
// exercised without a terminal. ASCII only, by design: the TUI must look the
// same on every console codepage.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace tui {

// Sparkline density ramp, low -> high.
inline const char *sparkRamp() { return " .:-=+*#%@"; }

// Oldest sample on the left, newest on the right. Values are scaled against
// the visible window's own min/max, so a quiet series still shows its shape.
// Fewer samples than `width` leaves the left side blank. Always returns
// exactly `width` characters (empty string for width <= 0).
inline std::string sparkline(const std::vector<double> &samples, int width) {
    if (width <= 0) return std::string();
    std::string out(static_cast<size_t>(width), ' ');
    const int n = static_cast<int>(samples.size());
    if (n == 0) return out;

    const int start = std::max(0, n - width);      // window on the newest part
    const int count = n - start;
    const int pad   = std::max(0, width - count);  // right-align the newest

    double lo = samples[static_cast<size_t>(start)];
    double hi = lo;
    for (int i = start; i < n; ++i) {
        lo = std::min(lo, samples[static_cast<size_t>(i)]);
        hi = std::max(hi, samples[static_cast<size_t>(i)]);
    }

    const char *ramp = sparkRamp();
    const double span = hi - lo;
    for (int c = 0; c < count; ++c) {
        const double v = samples[static_cast<size_t>(start + c)];
        double t = (span > 1e-12) ? (v - lo) / span : 0.5;
        t = std::min(1.0, std::max(0.0, t));
        const int r = static_cast<int>(t * 9.0 + 0.5);
        out[static_cast<size_t>(pad + c)] = ramp[std::min(9, std::max(0, r))];
    }
    return out;
}

// One detection box in capture-frame pixels, origin = capture crop top-left.
struct Box {
    float x = 0.f, y = 0.f, w = 0.f, h = 0.f, conf = 0.f;
    int label = -1;
};

// The detector crops the frame around the screen centre, so the frame centre is
// the crosshair. Positions below are centre-relative, in [-1, 1].
inline float boxCenterNX(const Box &b, float frameW) {
    return frameW > 0.f ? (b.x + b.w * 0.5f - frameW * 0.5f) / (frameW * 0.5f) : 0.f;
}
inline float boxCenterNY(const Box &b, float frameH) {
    return frameH > 0.f ? (b.y + b.h * 0.5f - frameH * 0.5f) / (frameH * 0.5f) : 0.f;
}

// Smallest distance from the crosshair to a box centre.
// Returns -1 when there is no box to measure.
inline float nearestBoxDistance(const std::vector<Box> &boxes, float frameW, float frameH) {
    float best = -1.f;
    for (const Box &b : boxes) {
        const float nx = boxCenterNX(b, frameW);
        const float ny = boxCenterNY(b, frameH);
        const float d = std::sqrt(nx * nx + ny * ny);
        if (best < 0.f || d < best) best = d;
    }
    return best;
}

// True when the crosshair (centre of the frame) is inside any box.
inline bool crosshairInsideBox(const std::vector<Box> &boxes, float frameW, float frameH) {
    if (frameW <= 0.f || frameH <= 0.f) return false;
    for (const Box &b : boxes) {
        const float x0 = (b.x - frameW * 0.5f) / (frameW * 0.5f);
        const float x1 = (b.x + b.w - frameW * 0.5f) / (frameW * 0.5f);
        const float y0 = (b.y - frameH * 0.5f) / (frameH * 0.5f);
        const float y1 = (b.y + b.h - frameH * 0.5f) / (frameH * 0.5f);
        if (x0 <= 0.f && 0.f <= x1 && y0 <= 0.f && 0.f <= y1) return true;
    }
    return false;
}

// ASCII map of the detection boxes around the crosshair. The capture frame
// spans the whole canvas; [-1, 1] on both axes. Rows are returned top (frame
// top) first, each exactly `cols` wide.
inline std::vector<std::string> renderBoxMap(const std::vector<Box> &boxes,
                                             float frameW, float frameH,
                                             int cols, int rows) {
    std::vector<std::string> canvas;
    if (cols <= 0 || rows <= 0) return canvas;
    canvas.assign(static_cast<size_t>(rows), std::string(static_cast<size_t>(cols), ' '));

    const int cx = cols / 2;
    const int cy = rows / 2;
    for (int r = 0; r < rows; ++r) canvas[static_cast<size_t>(r)][static_cast<size_t>(cx)] = '.';
    for (int c = 0; c < cols; ++c) canvas[static_cast<size_t>(cy)][static_cast<size_t>(c)] = '.';

    auto toCol = [cols](float nx) {
        const float t = (nx + 1.f) * 0.5f;
        const int c = static_cast<int>(t * static_cast<float>(cols - 1) + 0.5f);
        return std::min(cols - 1, std::max(0, c));
    };
    auto toRow = [rows](float ny) {
        const float t = (ny + 1.f) * 0.5f;
        const int r = static_cast<int>(t * static_cast<float>(rows - 1) + 0.5f);
        return std::min(rows - 1, std::max(0, r));
    };

    for (const Box &b : boxes) {
        if (frameW <= 0.f || frameH <= 0.f) continue;
        const int c0 = toCol((b.x - frameW * 0.5f) / (frameW * 0.5f));
        const int c1 = toCol((b.x + b.w - frameW * 0.5f) / (frameW * 0.5f));
        const int r0 = toRow((b.y - frameH * 0.5f) / (frameH * 0.5f));
        const int r1 = toRow((b.y + b.h - frameH * 0.5f) / (frameH * 0.5f));

        if (c0 == c1 || r0 == r1) {  // collapsed: a single marker
            canvas[static_cast<size_t>(r0)][static_cast<size_t>(c0)] = '+';
            continue;
        }
        for (int c = c0; c <= c1; ++c) {
            canvas[static_cast<size_t>(r0)][static_cast<size_t>(c)] = '-';
            canvas[static_cast<size_t>(r1)][static_cast<size_t>(c)] = '-';
        }
        for (int r = r0; r <= r1; ++r) {
            canvas[static_cast<size_t>(r)][static_cast<size_t>(c0)] = '|';
            canvas[static_cast<size_t>(r)][static_cast<size_t>(c1)] = '|';
        }
        canvas[static_cast<size_t>(r0)][static_cast<size_t>(c0)] = '+';
        canvas[static_cast<size_t>(r0)][static_cast<size_t>(c1)] = '+';
        canvas[static_cast<size_t>(r1)][static_cast<size_t>(c0)] = '+';
        canvas[static_cast<size_t>(r1)][static_cast<size_t>(c1)] = '+';

        // Class digit just inside the top-left corner when there is room.
        if (b.label >= 0 && (c1 - c0) >= 4 && (r1 - r0) >= 2) {
            canvas[static_cast<size_t>(r0 + 1)][static_cast<size_t>(c0 + 1)] =
                static_cast<char>('0' + (b.label % 10));
        }
    }

    // The crosshair is drawn last so it always stays visible through a box.
    canvas[static_cast<size_t>(cy)][static_cast<size_t>(cx)] = '+';
    return canvas;
}

} // namespace tui
