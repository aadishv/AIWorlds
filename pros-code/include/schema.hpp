#pragma once

#include "nlohmann/json.hpp"
#include <stdexcept>
#include <string>

struct Detection {
    double x;
    double y;
    double width;
    double height;
    int cls;
    double depth;

    /// Construct and validate
    Detection(double x_, double y_, double width_, double height_, int cls_,
              double depth_)
        : x(x_), y(y_), width(width_), height(height_), cls(cls_),
          depth(depth_) {
        if (x_ < 0.0 || x_ > 640.0)
            throw std::invalid_argument("Detection: x must be in [0,640]");
        if (y_ < 0.0 || y_ > 640.0)
            throw std::invalid_argument("Detection: y must be in [0,640]");
        if (width_ < 0.0 || width_ > 640.0)
            throw std::invalid_argument("Detection: width must be in [0,640]");
        if (height_ < 0.0 || height_ > 640.0)
            throw std::invalid_argument("Detection: height must be in [0,640]");
        if (cls_ < 0 || cls_ > 3)
            throw std::invalid_argument(
                "Detection: cls must be one of {0,1,2,3}");
        // depth has no range restriction
    }

    /// Load from a nlohmann::json object, rethrowing both JSON errors and
    /// validation errors
    static Detection from_json(const nlohmann::json &j) {
        try {
            double x = j.at("x").get<double>();
            double y = j.at("y").get<double>();
            double width = j.at("width").get<double>();
            double height = j.at("height").get<double>();
            int cls = j.at("cls").get<int>();
            double depth = j.at("depth").get<double>();

            return Detection{x, y, width, height, cls, depth};
        } catch (const nlohmann::json::exception &e) {
            // JSON parsing errors (missing key, wrong type, etc.)
            throw std::runtime_error(
                std::string("Detection::from_json JSON error: ") + e.what());
        }
        // any std::invalid_argument thrown by the constructor will propagate
    }
};
