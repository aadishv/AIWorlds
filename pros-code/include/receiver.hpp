#pragma once
#include <vector>
#include <string>
#include <optional>

namespace serial {
struct Detection {
    double x;
    double y;
    double w;
    double h;
    std::string cls;
    double depth;
    double conf;
};

struct Frame {
    std::string flag;
    std::vector<Detection> detections;
    std::vector<std::tuple<double, double, double>> poses;
};

std::optional<Frame> fetch_frame();
}
