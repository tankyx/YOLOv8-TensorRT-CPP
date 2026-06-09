#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <opencv2/opencv.hpp>
#include <thread>

#include "yolov8.h"

template <typename T> class SafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable cv;

public:
    // Push using perfect forwarding
    template <typename U> void push(U &&value) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(std::forward<U>(value));
        cv.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return !queue.empty(); });
        T value = std::move(queue.front());
        queue.pop();
        return value;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }
};

class DetectionQueue {
public:
    DetectionQueue() : moveThresholdPx_(5) {}
    explicit DetectionQueue(int moveThresholdPx) : moveThresholdPx_(moveThresholdPx) {}

    void setMoveThresholdPx(int threshold) {
        std::lock_guard<std::mutex> lock(mutex_);
        moveThresholdPx_ = threshold;
    }

    void push(const std::vector<Object> &detections) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (hasMovedMoreThanThreshold(detections, moveThresholdPx_)) {
            detections_ = detections;
        }
    }

    std::vector<Object> pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        return detections_;
    }

private:
    bool hasMovedMoreThanThreshold(const std::vector<Object> &newDetections, int threshold) {
        if (newDetections.size() != detections_.size()) {
            return true; // Different number of detections, consider it moved
        }
        for (size_t i = 0; i < newDetections.size(); ++i) {
            int dx = newDetections[i].rect.x - detections_[i].rect.x;
            int dy = newDetections[i].rect.y - detections_[i].rect.y;
            if (std::abs(dx) > threshold || std::abs(dy) > threshold) {
                return true;
            }
        }
        return false;
    }

    std::vector<Object> detections_;
    std::mutex mutex_;
    int moveThresholdPx_;
};

class LatencyQueue {
public:
    void push(long long latency) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::steady_clock::now();
        queue_.push_back({latency, now});

        // Remove old entries
        while (!queue_.empty() && std::chrono::duration_cast<std::chrono::milliseconds>(now - queue_.front().timestamp).count() > 500) {
            queue_.pop_front();
        }
    }

    double getAverageLatency() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return 0.0;

        double totalLatency = 0.0;
        int count = 0;
        for (const auto &entry : queue_) {
            totalLatency += entry.latency;
            ++count;
        }
        return totalLatency / count;
    }

    double getMinLatency() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return 0.0;
        double minVal = queue_.front().latency;
        for (const auto &entry : queue_) {
            if (entry.latency < minVal) minVal = entry.latency;
        }
        return minVal;
    }

    double getMaxLatency() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return 0.0;
        double maxVal = queue_.front().latency;
        for (const auto &entry : queue_) {
            if (entry.latency > maxVal) maxVal = entry.latency;
        }
        return maxVal;
    }

private:
    struct LatencyEntry {
        long long latency;
        std::chrono::steady_clock::time_point timestamp;
    };

    std::deque<LatencyEntry> queue_;
    std::mutex mutex_;
};

class FrameQueue {
public:
    // Take by value so callers can either copy (cv::Mat shallow-ref) or std::move into us;
    // either way the queue ends up owning a Mat handle without an unnecessary refcount bump.
    void push(cv::Mat frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!queue_.empty()) {
            queue_.pop(); // Drop the old frame; we only ever surface the latest one.
        }
        queue_.push(std::move(frame));
        cond_var_.notify_one();
    }

    cv::Mat pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this] { return !queue_.empty(); });
        cv::Mat frame = std::move(queue_.front());
        queue_.pop();
        return frame;
    }

private:
    std::queue<cv::Mat> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
};

// c39c: GPU-side counterpart to FrameQueue, used by the DXGI/CUDA-interop capture path.
// cv::cuda::GpuMat refcounts its underlying device buffer, so push/pop is shallow.
class GpuFrameQueue {
public:
    void push(cv::cuda::GpuMat frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!queue_.empty()) {
            queue_.pop();
        }
        queue_.push(std::move(frame));
        cond_var_.notify_one();
    }

    cv::cuda::GpuMat pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this] { return !queue_.empty(); });
        cv::cuda::GpuMat frame = std::move(queue_.front());
        queue_.pop();
        return frame;
    }

private:
    std::queue<cv::cuda::GpuMat> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
};