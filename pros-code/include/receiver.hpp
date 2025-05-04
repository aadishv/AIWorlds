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

std::optional<Frame> fetch_frame();
extern std::optional<Frame> most_recent_frame;
}
