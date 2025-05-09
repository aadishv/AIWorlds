#pragma once
#include <vector>
#include <string>
#include <optional>
#include <tuple>

namespace serial {
    struct Detection {
        double x;
        double y;
        double z;
        std::string cls;
    };

    struct Frame {
        std::string flag;
        std::vector<Detection> detections;
        std::tuple<double, double, double> pose;
    };

    extern std::optional<Frame> most_recent_frame;

    void task_send();
    void task_receive();
} // namespace serial
